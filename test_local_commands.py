import asyncio
from concurrent.futures import Future
from datetime import datetime
from contextlib import closing
import queue
import threading
from types import SimpleNamespace as NS
import unittest
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import patch
from local_commands import LocalRouter, LocalSpeechStream, Result
from hey_stars import Generation, Inference, Turn
from stars_panel import PanelState, PanelHandler, StarsPanel

class LocalTests(unittest.TestCase):
    def test_exact_intents(self):
        router=LocalRouter()
        for text,name in [('What time is it?','time'),('Open Chrome','chrome'),('increase volume','volume_up'),('show my tasks','tasks'),('stop speaking','stop')]:
            self.assertEqual(router.match(text).name,name)
    def test_task_phrases_do_not_guess_misheard_words(self):
        router=LocalRouter()
        for text in ['list tasks','list my tasks','task list','read my tasks','show my task']:
            self.assertEqual(router.match(text).name,'tasks')
        for text in ['so my task','zola task','the waiters','increase']:
            self.assertIsNone(router.match(text))
    def test_task_button_reads_database_without_replacing_voice_reply(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / '.tmp') as folder:
            path=Path(folder)/'tasks.sqlite3'
            with closing(sqlite3.connect(path)) as db:
                db.execute('CREATE TABLE tasks (id INTEGER PRIMARY KEY, title TEXT, status TEXT)')
                db.executemany('INSERT INTO tasks(title,status) VALUES (?,?)',
                               [('Submit assignment','pending'),('Old task','done')])
                db.commit()
            panel=object.__new__(StarsPanel)
            panel.local_router=LocalRouter(path)
            panel.tasks_busy=threading.Event()
            panel.stop=threading.Event()
            panel.handler=PanelHandler()
            panel.show_tasks()
            message=panel.handler.events.get(timeout=3)
            state=PanelState()
            state.consume('STARS (queued for speech): Existing answer.')
            original_status=state.status
            state.consume(message)
            self.assertIn('Submit assignment',state.tasks)
            self.assertNotIn('Old task',state.tasks)
            self.assertEqual(state.answer,'Existing answer.')
            self.assertEqual(state.status,original_status)
            state.consume('Listening for command (generation 2)')
            self.assertIn('Submit assignment',state.tasks)
    def test_unrelated_negated_or_quoted_commands_do_not_execute(self):
        router=LocalRouter()
        for text in ['do not open chrome','explain how to open chrome','what does open chrome mean','open chrome and delete files','what is on my screen']:
            self.assertIsNone(router.match(text))
    def make_engine(self,text,confidence=.99,local_only=False):
        output=NS(sentences=queue.Queue(),interrupt=lambda:None)
        engine=Inference('',Generation(),threading.Event(),queue.Queue(),output,NS(reserve=lambda n:self.fail('Quota gate must not be called')),local_recognizer=NS(transcribe=lambda b:(text,confidence)),local_router=LocalRouter(),local_only=local_only)
        return engine,output
    def test_local_command_bypasses_api_and_quota(self):
        engine,output=self.make_engine('what time is it')
        self.assertTrue(asyncio.run(engine.handle_local(Turn(0,b'',b'',0))))
        self.assertIn('It is',output.sentences.get()[1])
    def test_low_confidence_action_does_not_execute(self):
        engine,output=self.make_engine('open chrome',.5)
        with patch.object(engine.local_router,'execute',side_effect=AssertionError('Action must not execute')):
            self.assertTrue(asyncio.run(engine.handle_local(Turn(0,b'',b'',0))))
        self.assertEqual(output.sentences.get_nowait()[1],'Please repeat the command clearly.')
    def test_unclear_local_intents_never_execute_or_fall_through_to_cloud(self):
        for text in ['increase volume','decrease volume','show my tasks','what time is it']:
            with self.subTest(text=text):
                engine,output=self.make_engine(text,.5,local_only=False)
                with patch.object(engine.local_router,'execute',side_effect=AssertionError('Unclear action executed')):
                    self.assertTrue(asyncio.run(engine.handle_local(Turn(0,b'',b'',0))))
                self.assertEqual(output.sentences.get_nowait()[1],'Please repeat the command clearly.')
    def test_known_misrecognitions_do_not_become_actions(self):
        for text in ['increase','and chrome','oh been chrome','will burn chrome','so my task','zola task']:
            with self.subTest(text=text):
                engine,output=self.make_engine(text,1.0,local_only=True)
                with patch.object(engine.local_router,'execute',side_effect=AssertionError('Misrecognition executed')):
                    self.assertTrue(asyncio.run(engine.handle_local(Turn(0,b'',b'',0))))
                self.assertEqual(output.sentences.get_nowait()[1],'Please try a local command.')
    def test_offline_unknown_never_requests_gemini(self):
        engine,output=self.make_engine('describe my screen',local_only=True)
        with self.assertLogs('stars',level='INFO') as logs:
            self.assertTrue(asyncio.run(engine.handle_local(Turn(0,b'',b'',0))))
        self.assertEqual(output.sentences.get()[1],'Please try a local command.')
        self.assertTrue(any('No Gemini request was made' in line for line in logs.output))
    def test_stale_command_does_not_execute(self):
        engine,output=self.make_engine('open chrome')
        engine.epoch.advance()
        with patch.object(engine.local_router,'execute',side_effect=AssertionError('Stale action must not execute')):
            self.assertTrue(asyncio.run(engine.handle_local(Turn(0,b'',b'',0))))
    def test_stop_is_silent(self):
        engine,output=self.make_engine('stop speaking')
        self.assertTrue(asyncio.run(engine.handle_local(Turn(0,b'',b'',0))))
        self.assertTrue(output.sentences.empty())
    def test_streaming_final_result_avoids_batch_decode(self):
        engine,output=self.make_engine('must not decode again')
        result=Future()
        result.set_result(('what time is it',.99))
        with patch.object(engine.local_recognizer,'transcribe',side_effect=AssertionError('Repeated decoding')):
            self.assertTrue(asyncio.run(engine.handle_local(Turn(0,b'',b'',0,result))))
        self.assertIn('It is',output.sentences.get()[1])
    def test_streaming_failure_never_executes_or_uses_cloud(self):
        engine,output=self.make_engine('open chrome')
        result=Future()
        result.set_exception(RuntimeError('Missing audio'))
        with patch.object(engine.local_router,'execute',side_effect=AssertionError('Incomplete command')):
            self.assertTrue(asyncio.run(engine.handle_local(Turn(0,b'',b'',0,result))))
        self.assertTrue(output.sentences.empty())
    def test_stream_collects_only_final_segments_and_confidence(self):
        accepted=threading.Event()
        class Decoder:
            def SetWords(self,enabled): pass
            def AcceptWaveform(self,block):
                accepted.set()
                return True
            def Result(self): return '{"text":"do not", "result":[{"conf":0.94}]}'
            def FinalResult(self): return '{"text":"open chrome", "result":[{"conf":0.91}]}'
        with patch('vosk.KaldiRecognizer',return_value=Decoder()):
            stream=LocalSpeechStream(NS(model=None,lock=threading.Lock()))
            try:
                stream.feed(b'pcm')
                self.assertTrue(accepted.wait(timeout=3))
                self.assertFalse(stream.result.done())
                self.assertEqual(stream.finish().result(timeout=3),('do not open chrome',.91))
                self.assertIsNone(LocalRouter().match(stream.result.result()[0]))
            finally:
                stream.cancel()
                stream.worker.join(timeout=3)
    def test_stream_backpressure_invalidates_whole_utterance(self):
        owner=NS(model=None,lock=threading.Lock())
        owner.lock.acquire()
        stream=LocalSpeechStream(owner)
        try:
            for _ in range(33): stream.feed(b'pcm')
            with self.assertRaisesRegex(RuntimeError,'queue full'):
                stream.finish().result(timeout=3)
        finally:
            stream.cancel()
            owner.lock.release()
            stream.worker.join(timeout=3)
        self.assertFalse(stream.worker.is_alive())
    def test_stream_cancellation_releases_worker(self):
        owner=NS(model=None,lock=threading.Lock())
        owner.lock.acquire()
        stream=LocalSpeechStream(owner)
        stream.cancel()
        owner.lock.release()
        stream.worker.join(timeout=3)
        self.assertFalse(stream.worker.is_alive())
        self.assertTrue(stream.result.cancelled())

if __name__=='__main__': unittest.main()
