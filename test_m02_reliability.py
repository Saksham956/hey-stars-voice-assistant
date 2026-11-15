"""M02-R regression checks: no real microphone, network, desktop or Gemini."""
import importlib.util
import asyncio
import json
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


class ReliabilityTests(unittest.TestCase):
    def test_powershell_wrapped_log_keeps_success_and_rejection_distinct(self):
        from analyze_stars_log import parse_log, summarize
        log = '''powershell : 2000-01-01 02:29:45,026 INFO Selected input device=1
At line:1 char:1
+ powershell -ExecutionPolicy Bypass
    + FullyQualifiedErrorId : NativeCommandError
2000-01-01 02:29:56,518 INFO Listening for command (generation 1)
2000-01-01 02:30:00,108 INFO YOU (local transcript): what is it
2000-01-01 02:30:00,108 INFO Local timing generation=1 recognition_wait_seconds=0.060160 match_seconds=0.000442
execution_seconds=0.000000 outcome=unmatched intent=none
2000-01-01 02:30:00,570 INFO TTFA generation=1 end-of-speech-to-first-PCM=0.848s (device latency excluded)
2000-01-01 02:30:07,272 INFO Listening for command (generation 2)
2000-01-01 02:30:09,717 INFO YOU (local transcript): what time is it
2000-01-01 02:30:09,718 INFO Local timing generation=2 recognition_wait_seconds=0.015668 match_seconds=0.000015
execution_seconds=0.000944 outcome=handled intent=time
2000-01-01 02:30:09,718 INFO Local action generation=2 end-of-speech-to-result=0.393s
2000-01-01 02:30:10,231 INFO TTFA generation=2 end-of-speech-to-first-PCM=0.878s (device latency excluded)'''
        rows = parse_log(log)
        result = summarize(rows)
        self.assertEqual(rows[0]['outcome'],'unmatched')
        self.assertEqual(rows[1]['intent'],'time')
        self.assertEqual(rows[1]['execution_seconds'],.000944)
        self.assertEqual(result['completed_with_pcm_or_silent_stop'],1)
        self.assertEqual(result['failure_or_missing_response_count'],1)
        self.assertEqual(result['first_pcm_samples'],1)
        self.assertEqual(result['median_first_pcm_seconds'],.878)

    def input_devices(self, items, default=0):
        from unittest.mock import Mock
        return NS(query_devices=lambda:items,default=NS(device=[default,0]),check_input_settings=Mock())

    def test_named_headset_does_not_follow_changed_default_mic(self):
        from hey_stars import resolve_input_device
        sd=self.input_devices([{'name':'Laptop mic','hostapi':0,'max_input_channels':1},
                               {'name':'Headset (MOTIF II A.N.C.)','hostapi':0,'max_input_channels':1}])
        self.assertEqual(resolve_input_device(sd,name='Headset (MOTIF II A.N.C.)'),1)
        sd.check_input_settings.assert_called_once_with(device=1,samplerate=16000,channels=1,dtype='int16')

    def test_missing_headset_never_silently_selects_laptop_mic(self):
        from hey_stars import resolve_input_device
        sd=self.input_devices([{'name':'Laptop mic','hostapi':0,'max_input_channels':1}])
        with self.assertRaisesRegex(ValueError,'unavailable'):
            resolve_input_device(sd,name='Headset (MOTIF II A.N.C.)')
        sd.check_input_settings.assert_not_called()

    def test_device_number_is_explicit_and_output_only_device_rejected(self):
        from hey_stars import resolve_input_device
        sd=self.input_devices([{'name':'Laptop mic','hostapi':0,'max_input_channels':1},
                               {'name':'Speakers','hostapi':0,'max_input_channels':0}])
        self.assertEqual(resolve_input_device(sd,index=0),0)
        with self.assertRaisesRegex(ValueError,'not an available microphone'):
            resolve_input_device(sd,index=1)

    def test_named_headset_keeps_default_host_api_profile(self):
        from hey_stars import resolve_input_device
        sd=self.input_devices([{'name':'Other','hostapi':0,'max_input_channels':1},
                               {'name':'Headset','hostapi':2,'max_input_channels':1},
                               {'name':'Headset','hostapi':0,'max_input_channels':1}])
        self.assertEqual(resolve_input_device(sd,name='Headset'),2)

    def test_duplicate_microphones_in_same_host_require_explicit_selection(self):
        from hey_stars import resolve_input_device
        sd=self.input_devices([{'name':'Headset','hostapi':0,'max_input_channels':1},
                               {'name':'Headset','hostapi':0,'max_input_channels':1}])
        with self.assertRaisesRegex(ValueError,'ambiguous'):
            resolve_input_device(sd,name='Headset')

    def test_endpoint_no_change_is_an_error_not_success(self):
        from windows_volume import change_volume
        with patch('windows_volume.EndpointVolume') as factory:
            endpoint=factory.return_value.__enter__.return_value
            endpoint.level.side_effect=[.5,.5]
            with self.assertRaisesRegex(OSError,'did not change'):
                change_volume('volume_up')
            endpoint.step.assert_called_once_with('volume_up')
            factory.return_value.__exit__.assert_called_once()

    def test_endpoint_limit_does_not_step_and_reports_limit(self):
        from windows_volume import change_volume
        with patch('windows_volume.EndpointVolume') as factory:
            endpoint=factory.return_value.__enter__.return_value
            endpoint.level.return_value=1.
            endpoint.muted.return_value=False
            result=change_volume('volume_up')
            self.assertTrue(result.at_limit)
            endpoint.step.assert_not_called()

    def test_endpoint_cancellation_before_step_has_no_side_effect(self):
        from windows_volume import change_volume
        from unittest.mock import Mock
        current=Mock(side_effect=[True,False])
        with patch('windows_volume.EndpointVolume') as factory:
            endpoint=factory.return_value.__enter__.return_value
            endpoint.level.return_value=.5
            self.assertIsNone(change_volume('volume_up',is_current=current))
            endpoint.step.assert_not_called()
            factory.return_value.__exit__.assert_called_once()

    def test_recognition_diagnostics_are_bounded_and_single_line(self):
        from hey_stars import diagnostic_text
        value=diagnostic_text('open chrome\n'+'a'*1000)
        self.assertEqual(len(value),240)
        self.assertNotIn('\n',value)

    def test_volume_aliases_execute_only_one_checked_endpoint_step(self):
        from local_commands import LocalRouter
        from windows_volume import VolumeChange
        router = LocalRouter()
        for phrase, intent in [('increase volume','volume_up'),
                               ('turn up the volume','volume_up'),
                               ('decrease volume','volume_down'),
                               ('turn down the volume','volume_down')]:
            result=VolumeChange(.5,.52 if intent=='volume_up' else .48,False)
            with self.subTest(phrase=phrase), patch('windows_volume.change_volume',return_value=result) as change:
                matched = router.match(phrase)
                self.assertEqual(matched.name, intent)
                router.execute(matched)
                change.assert_called_once()
                self.assertEqual(change.call_args.args,(intent,))

    def test_unsupported_mute_and_unrelated_phrases_do_not_match(self):
        from local_commands import LocalRouter
        router = LocalRouter()
        for phrase in ('mute','unmute','mute volume','unmute volume','how are you',
                       'explain what time is it','do not decrease volume',
                       'so my task','zola task','the waiters'):
            with self.subTest(phrase=phrase):
                self.assertIsNone(router.match(phrase))

    def test_pending_media_lookup_does_not_block_next_local_turn(self):
        from hey_stars import Generation, Inference, Microphone, Turn
        from local_commands import LocalRouter
        from youtube_media import MediaLookupError
        import numpy as np
        entered, release = threading.Event(), threading.Event()
        def lookup(query):
            entered.set()
            if not release.wait(3): raise TimeoutError('Test lookup release missing')
            raise MediaLookupError('simulated offline')
        router = LocalRouter()
        epoch, stop, turns = Generation(), threading.Event(), queue.Queue()
        output = NS(sentences=queue.Queue(), interrupt=lambda: None)
        recognizer = NS(transcribe=lambda wav: ('play test song' if wav == b'media' else 'what time is it', .99))
        engine = Inference('',epoch,stop,turns,output,
                           NS(reserve=lambda n:self.fail('Local turns used quota')),
                           local_recognizer=recognizer,local_router=router,local_only=True)
        async def scenario():
            task = asyncio.create_task(engine.loop())
            try:
                turns.put(Turn(0,b'media',b'',0))
                async with asyncio.timeout(2):
                    while not entered.is_set(): await asyncio.sleep(.01)
                    generation = epoch.advance()
                    # A real callback can enqueue without waiting for the lookup.
                    mic = Microphone(None,None,None,None,None,None,None,None,None,None,.65)
                    mic.callback(np.zeros((512,1),dtype=np.int16),512,None,False)
                    self.assertEqual(mic.blocks.qsize(),1)
                    turns.put(Turn(generation,b'time',b'',0))
                    while output.sentences.empty(): await asyncio.sleep(.01)
                self.assertIn('It is', output.sentences.get_nowait()[1])
                self.assertFalse(release.is_set())
            finally:
                release.set(); stop.set()
                await asyncio.wait_for(task,2)
        with patch('youtube_media.find_video',side_effect=lookup), patch.object(router,'_open_chrome') as launch:
            asyncio.run(scenario())
            launch.assert_not_called()

    def test_latency_summary_counts_failures_and_excludes_rejection_audio(self):
        from analyze_stars_log import parse_log, summarize
        text = '''Listening for command (generation 1)
YOU (local transcript): what time is it
Local timing generation=1 recognition_wait_seconds=0.030000 match_seconds=0.000020 execution_seconds=0.001000 outcome=handled intent=time
TTFA generation=1 end-of-speech-to-first-PCM=0.800s (device latency excluded)
Listening for command (generation 2)
Local timing generation=2 recognition_wait_seconds=0.020000 match_seconds=0.000010 execution_seconds=0.000000 outcome=unmatched intent=none
TTFA generation=2 end-of-speech-to-first-PCM=0.600s (device latency excluded)
Listening for command (generation 3)
Local timing generation=3 recognition_wait_seconds=0.030000 match_seconds=0.000010 execution_seconds=0.001000 outcome=handled intent=stop'''
        rows = parse_log(text)
        summary = summarize(rows, expected_trials=4)
        self.assertEqual(summary['median_first_pcm_seconds'], .8)
        self.assertEqual(summary['first_pcm_samples'], 1)
        self.assertEqual(summary['failure_or_missing_response_count'], 2)
        self.assertEqual(summary['unobserved_attempts'], 1)
        self.assertEqual(rows[0]['transcript'], 'what time is it')

    def test_connection_module_import_is_noninteractive(self):
        path = Path(__file__).with_name('test_connection.py')
        spec = importlib.util.spec_from_file_location('connection_import_check', path)
        module = importlib.util.module_from_spec(spec)
        with patch('getpass.getpass', side_effect=AssertionError('Import prompted')):
            spec.loader.exec_module(module)
        self.assertTrue(callable(module.main))

    def test_timeout_retains_active_test_and_partial_counts(self):
        from run_offline_tests import run_bounded
        class Process:
            returncode = None
            killed = False
            def wait(self, timeout):
                if not self.killed:
                    raise subprocess.TimeoutExpired('test', timeout)
                self.returncode = -1
                return -1
            def kill(self): self.killed = True
            def poll(self): return self.returncode
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory/'progress.json').write_text(json.dumps({
                'discovered': 3, 'started': 2, 'active_test': 'blocked_test',
                'complete': False, 'outcomes': [{'test':'done', 'outcome':'passed'}]}))
            process = Process()
            with patch('run_offline_tests.subprocess.Popen', return_value=process):
                summary = run_bounded(['placeholder'], .1, directory)
            self.assertTrue(process.killed)
            self.assertEqual((summary['passed'], summary['timed_out'], summary['not_run']), (1,1,1))
            self.assertEqual(summary['active_test'], 'blocked_test')


if __name__ == '__main__':
    unittest.main()
