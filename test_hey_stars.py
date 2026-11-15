"""Offline checks: no models, microphone, network, or API key required."""
import asyncio
from datetime import timezone
import io
import logging
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
import hey_stars as h

class Tests(unittest.TestCase):
    def test_console_backpressure_never_blocks_producer(self):
        from stars_logging import AsyncConsoleHandler
        entered,release=threading.Event(),threading.Event()
        def slow_sink(record):
            entered.set()
            release.wait(timeout=3)
        handler=AsyncConsoleHandler(sink=NS(emit=slow_sink,setFormatter=lambda f:None),capacity=1)
        record=logging.LogRecord('stars',logging.INFO,__file__,0,'test',(),None)
        try:
            handler.emit(record)
            self.assertTrue(entered.wait(timeout=3))
            handler.emit(record)
            handler.emit(record)
            self.assertEqual(handler.dropped_records,1)
            self.assertEqual(handler.records.qsize(),1)
        finally:
            release.set()
            handler.close()
        self.assertFalse(handler.worker.is_alive())
    def test_panel_state(self):
        from stars_panel import PanelState
        state=PanelState()
        state.consume('Listening for command (generation 1)')
        state.consume('YOU (Gemini transcript): Explain this screen')
        state.consume('STARS (queued for speech): A terminal is open.')
        self.assertEqual(state.heard,'Explain this screen')
        self.assertEqual(state.answer,'A terminal is open.')
        state.consume('Listening for command (generation 2)')
        self.assertEqual(state.answer,'')
        state.consume('Gemini request failed (ReadTimeout, code=None)')
        self.assertIn('failed',state.status)
    def test_clause_chunk_boundaries(self):
        c=h.Clauses()
        self.assertEqual(c.feed('Hello'),[])
        self.assertEqual(c.feed(', world! Next'),['Hello,','world!'])
        self.assertEqual(c.finish(),['Next'])
    def test_decimal_split_tokens(self):
        c=h.Clauses()
        self.assertEqual(c.feed('Model v0.'), [])
        self.assertEqual(c.feed('1 is ready,'), ['Model v0.1 is ready,'])
        self.assertEqual(c.feed(' value 3.14!'), ['value 3.14!'])
    def test_clause_colons_preserve_split_times_urls_and_paths(self):
        c=h.Clauses()
        self.assertEqual(c.feed('Time 4:'),[])
        self.assertEqual(c.feed('01; next:'),['Time 4:01;','next:'])
        c=h.Clauses()
        self.assertEqual(c.feed('Visit https:'),[])
        self.assertEqual(c.feed('//example.com, then C:'),['Visit https://example.com,'])
        self.assertEqual(c.feed('\\Apps;'),['then C:\\Apps;'])
    def test_short_opening_chunk(self):
        c=h.Clauses()
        text='You have a command prompt open showing log messages for your voice assistant project,'
        parts=c.feed(text)+c.finish()
        self.assertEqual(len(parts[0].split()),8)
        self.assertEqual(' '.join(parts),text)
    def test_long_clause_bounded(self):
        c=h.Clauses()
        self.assertTrue(c.feed('word '*50))
        self.assertLess(len(c.buffer),160)
    @patch('hey_stars.ZoneInfo', return_value=timezone.utc)
    def test_persistent_daily_and_rolling_quota(self, zone):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / '.tmp') as folder:
            path=Path(folder)/'usage.db'
            gate=h.RateGate(path,1,8192,2)
            self.assertTrue(gate.reserve(8192,now=1800000000))
            self.assertFalse(gate.reserve(1,now=1800000001))
            gate=h.RateGate(path,1,8192,2)
            self.assertTrue(gate.reserve(8192,now=1800000062))
            self.assertFalse(gate.reserve(1,now=1800000124))
    @patch('hey_stars.ZoneInfo', return_value=timezone.utc)
    def test_tpm_reservation(self, zone):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / '.tmp') as folder:
            gate=h.RateGate(Path(folder)/'usage.db',10,8191,10)
            self.assertFalse(gate.reserve(8192))
    def test_barge_in_discards_queued_and_current_audio(self):
        output=object.__new__(h.SpeechOutput)
        output.epoch=h.Generation()
        output.stop=NS(is_set=lambda:False)
        output.sentences=queue.Queue(); output.audio=queue.Queue()
        output.current=(0,np.ones(1000,dtype=np.float32),0)
        output.position=10; output.metrics=queue.SimpleQueue(); output.first_audio=set()
        output.audio.put((0,np.ones(20,dtype=np.float32),0))
        output.sentences.put((0,'stale',0))
        output.epoch.advance(); output.interrupt()
        out=np.ones((480,1),dtype=np.float32)
        output.callback(out,480,None,None)
        self.assertTrue(np.all(out==0))
        self.assertTrue(output.audio.empty() and output.sentences.empty())
    def test_gemini_stream_config_and_early_speech(self):
        epoch=h.Generation(); stop=h.threading.Event()
        output=NS(sentences=queue.Queue())
        inference=h.Inference('not-a-key',epoch,stop,queue.Queue(),output,NS(reserve=lambda n:True))
        captured={}; closed=[]
        class Stream:
            def __aiter__(self): return self.responses()
            async def responses(self):
                yield NS(text='Hello,')
                self_test.assertFalse(output.sentences.empty())
                yield NS(text=' world!')
            async def aclose(self): closed.append(True)
        async def generate(**kw): captured.update(kw); return Stream()
        self_test=self
        client=NS(aio=NS(models=NS(generate_content_stream=generate)))
        asyncio.run(inference.generate(client,h.Turn(0,b'wav',b'jpeg',0)))
        self.assertEqual(captured['model'],'gemini-2.5-flash')
        self.assertEqual(captured['config'].thinking_config.thinking_budget,0)
        self.assertEqual([p.inline_data.mime_type for p in captured['contents'][0].parts if p.inline_data],['image/jpeg','audio/wav'])
        self.assertTrue(closed)
    def test_stale_synthesis_cannot_enqueue(self):
        epoch=h.Generation(); epoch.advance()
        q=queue.Queue()
        self.assertFalse(h.put_current(q,'stale',0,epoch,h.threading.Event()))
        self.assertTrue(q.empty())
    def test_microphone_callback_marks_driver_and_queue_overruns(self):
        mic=h.Microphone(None,None,None,None,None,None,None,None,None,None,.65)
        frame=np.zeros((512,1),dtype=np.int16)
        mic.callback(frame,512,None,True)
        self.assertTrue(mic.driver_overflow.is_set())
        self.assertFalse(mic.queue_overflow.is_set())
        mic.overflow.clear()
        mic.driver_overflow.clear()
        while not mic.blocks.full(): mic.callback(frame,512,None,False)
        mic.callback(frame,512,None,False)
        self.assertTrue(mic.overflow.is_set())
        self.assertTrue(mic.queue_overflow.is_set())
        self.assertFalse(mic.driver_overflow.is_set())
    def test_command_starting_immediately_after_beep_keeps_first_word(self):
        # A four-frame short command begins as the beep ends. The old fixed
        # three-frame skip drops 96ms, leaving too little speech to accept.
        from concurrent.futures import Future
        clock=[100.0]
        stop=threading.Event()
        turns=queue.Queue()
        fed=[]
        future=Future(); future.set_result(('open chrome',.99))
        stream=NS(feed=fed.append,finish=lambda:future,cancel=lambda:None)
        class Blocks(queue.Queue):
            def get(self,*args,**kwargs):
                try: item=super().get(block=False)
                except queue.Empty:
                    stop.set(); raise
                clock[0]=max(clock[0],item[0])
                return item
        wake=NS(models={'jarvis':None},reset=lambda:None,predict=lambda x:{'jarvis':1.0})
        class Vad:
            def reset_states(self): pass
            def __call__(self,audio,rate): return NS(item=lambda:float(np.any(audio)))
        mic=h.Microphone(wake,Vad(),NS(from_numpy=lambda x:x),NS(snapshot=lambda:b'jpeg'),
                         h.Generation(),NS(interrupt=lambda:None),turns,stop,None,None,.65,
                         NS(start_stream=lambda:stream))
        mic.blocks=Blocks(32)
        class Input:
            def __enter__(self):
                for index in range(3+4+11):
                    value=100 if 3 <= index < 7 else 0
                    mic.blocks.put_nowait((100.0+index*.032,np.full(512,value,dtype=np.int16)))
            def __exit__(self,*args): pass
        mic.sd=NS(InputStream=lambda **kw:Input(),query_devices=lambda *args:{'name':'test input'})
        def beep(*args): clock[0]=100.096
        with patch('winsound.Beep',side_effect=beep),patch('hey_stars.time.monotonic',side_effect=lambda:clock[0]):
            mic.run()
        self.assertEqual(turns.qsize(),1,'Post-beep first word was discarded')
        self.assertEqual(sum(np.any(np.frombuffer(p,dtype='<i2')) for p in fed),4)
    def test_input_clock_maps_buffered_audio_before_ready_boundary(self):
        # Audio delivered at 100s actually ended 68ms earlier. It must not be
        # mistaken for post-beep speech merely because the callback ran late.
        self.assertAlmostEqual(h.capture_end_time(100.,512,10.,10.1),99.932)
        self.assertEqual(h.capture_end_time(100.,512,0.,0.),100.)
        self.assertEqual(h.capture_end_time(100.,512,float('nan'),10.),100.)
        self.assertEqual(h.capture_end_time(100.,512,1.,100.),100.)
    def test_microphone_streams_preroll_and_keeps_trimmed_cloud_audio(self):
        from concurrent.futures import Future
        import io
        import wave
        stop=threading.Event()
        future=Future()
        fed=[]
        def finish():
            future.set_result(('what time is it',.99))
            return future
        stream=NS(feed=fed.append,finish=finish,cancel=lambda:self.fail('Completed capture was cancelled'))
        probabilities=iter([0,0]+[1]*4+[0]*11)
        class Vad:
            def reset_states(self): pass
            def __call__(self,audio,rate): return NS(item=lambda:next(probabilities))
        wake=NS(models={'jarvis':None},reset=lambda:None,predict=lambda x:{'jarvis':1.0})
        turns=queue.Queue()
        class Turns:
            def get_nowait(self): return turns.get_nowait()
            def put_nowait(self,turn):
                turns.put_nowait(turn)
                stop.set()
        mic=h.Microphone(wake,Vad(),NS(from_numpy=lambda x:x),NS(snapshot=lambda:b'jpeg'),
                         h.Generation(),NS(interrupt=lambda:None),Turns(),stop,None,None,.65,
                         NS(start_stream=lambda:stream))
        class Input:
            def __enter__(self):
                # Timestamped frames model three queued beep/reset frames,
                # followed by the actual command. All-at-once delivery must
                # not pretend every frame was captured before the beep.
                base=100.0
                for index in range(3+3+2+4+11):
                    mic.blocks.put_nowait((base+index*.032,np.full(512,index,dtype=np.int16)))
            def __exit__(self,*args): pass
        mic.sd=NS(InputStream=lambda **kw:Input(),query_devices=lambda *args:{'name':'test input'})
        # Keep the prior three cue frames before the new ready boundary.
        with patch('winsound.Beep'),patch('hey_stars.time.monotonic',return_value=100.192):
            mic.run()
        turn=turns.get_nowait()
        self.assertIs(turn.local_result,future)
        self.assertEqual(len(fed),17)
        with wave.open(io.BytesIO(turn.wav),'rb') as audio:
            self.assertEqual(audio.readframes(audio.getnframes()),b''.join(fed[:-9]))
    def test_paused_microphone_resets_once_and_drains_frames(self):
        from unittest.mock import Mock
        stop=threading.Event()
        wake=NS(models={'jarvis':None},reset=Mock())
        vad=NS(reset_states=Mock())
        mic=h.Microphone(wake,vad,None,None,None,None,None,stop,None,None,.65)
        mic.paused.set()
        class Blocks(queue.Queue):
            processed=0
            def get(self,*args,**kwargs):
                item=super().get(*args,**kwargs)
                self.processed+=1
                if self.processed==10: stop.set()
                return item
        mic.blocks=Blocks(32)
        class Input:
            def __enter__(self):
                for _ in range(10): mic.callback(np.zeros((512,1),dtype=np.int16),512,None,False)
            def __exit__(self,*args): pass
        mic.sd=NS(InputStream=lambda **kw:Input(),query_devices=lambda *args:{'name':'test input'})
        mic.run()
        wake.reset.assert_called_once()
        vad.reset_states.assert_called_once()
        self.assertTrue(mic.blocks.empty())
    def test_idle_overrun_reports_state_and_inference_timings(self):
        stop=threading.Event()
        wake=NS(models={'jarvis':None},reset=lambda:None)
        mic=h.Microphone(wake,NS(reset_states=lambda:None),None,None,None,None,None,stop,None,None,.65)
        class Blocks(queue.Queue):
            def get(self,*args,**kwargs):
                if kwargs.get('timeout'):
                    stop.set()
                    raise queue.Empty
                return super().get(*args,**kwargs)
        mic.blocks=Blocks(32)
        class Input:
            def __enter__(self):
                for _ in range(33): mic.callback(np.zeros((512,1),dtype=np.int16),512,None,False)
            def __exit__(self,*args): pass
        mic.sd=NS(InputStream=lambda **kw:Input(),query_devices=lambda *args:{'name':'test input'})
        with self.assertLogs('stars',level='WARNING') as logs: mic.run()
        self.assertTrue(any('state=idle' in line and 'audio processing queue full' in line and 'max wake=' in line for line in logs.output))

if __name__=='__main__': unittest.main()
