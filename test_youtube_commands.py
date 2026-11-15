"""Offline routing and YouTube lookup tests: no real browser, audio or network."""
import asyncio
import queue
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from hey_stars import Generation, Inference, Turn
from local_commands import Intent, LocalRouter
from youtube_media import MediaLookupError, Video, find_video


class YouTubeTests(unittest.TestCase):
    def test_spoken_phrases_and_titles(self):
        router=LocalRouter()
        phrases={
            'Open YouTube in Chrome': Intent('youtube'),
            'open you tube': Intent('youtube'),
            'play Believer by Imagine Dragons on YouTube': Intent('youtube_play','Believer by Imagine Dragons'),
            'play song Perfect by Ed Sheeran': Intent('youtube_play','Perfect by Ed Sheeran'),
            'open youtube on chrome and play lo fi music': Intent('youtube_play','lo fi music'),
            'search YouTube for Python tutorial': Intent('youtube_search','Python tutorial'),
            'find guitar lessons on youtube': Intent('youtube_search','guitar lessons'),
        }
        for text,intent in phrases.items():
            with self.subTest(text=text): self.assertEqual(router.match(text),intent)

    def test_negation_explanation_and_long_queries_do_not_route(self):
        router=LocalRouter()
        for text in ['do not play music','explain how to play music','do not open youtube',
                     'what does play music mean','play '+'a'*161]:
            self.assertIsNone(router.match(text))

    def test_homepage_opens_chrome_without_lookup(self):
        router=LocalRouter()
        with patch.object(router,'_open_chrome',return_value=True) as launch, patch('youtube_media.find_video') as lookup:
            result=router.execute(Intent('youtube'))
        launch.assert_called_once_with('https://www.youtube.com/')
        lookup.assert_not_called()
        self.assertEqual(result.message,'Opening YouTube.')

    def test_search_query_is_encoded_as_data(self):
        router=LocalRouter()
        query='AC/DC & friends #live "quoted"'
        with patch.object(router,'_open_chrome',return_value=True) as launch:
            router.execute(Intent('youtube_search',query))
        url=launch.call_args.args[0]
        self.assertEqual(urlparse(url).netloc,'www.youtube.com')
        self.assertEqual(parse_qs(urlparse(url).query),{'search_query':[query]})

    def test_direct_play_builds_watch_url(self):
        router=LocalRouter()
        with patch('youtube_media.find_video',return_value=Video('BaW_jenozKc','Test video')) as lookup, patch.object(router,'_open_chrome',return_value=True) as launch:
            result=router.execute(Intent('youtube_play','test video'))
        lookup.assert_called_once_with('test video')
        self.assertEqual(parse_qs(urlparse(launch.call_args.args[0]).query),{'v':['BaW_jenozKc'],'autoplay':['1']})
        self.assertIn('first YouTube match',result.message)

    def test_lookup_failure_opens_search_results_honestly(self):
        router=LocalRouter()
        with patch('youtube_media.find_video',side_effect=MediaLookupError('offline')), patch.object(router,'_open_chrome',return_value=True) as launch:
            result=router.execute(Intent('youtube_play','test song'))
        self.assertIn('/results?',launch.call_args.args[0])
        self.assertIn('Direct play is unavailable',result.message)

    def test_interruption_during_lookup_prevents_browser_launch(self):
        router=LocalRouter()
        current=threading.Event();current.set()
        def lookup(query):
            current.clear()
            return Video('BaW_jenozKc','Test video')
        with patch('youtube_media.find_video',side_effect=lookup), patch.object(router,'_open_chrome') as launch:
            result=router.execute(Intent('youtube_play','test song'),is_current=current.is_set)
        launch.assert_not_called()
        self.assertFalse(result.speak)

    def test_chrome_receives_argument_list_without_shell(self):
        router=LocalRouter()
        with patch('local_commands.Path.is_file',return_value=True), patch.dict('os.environ',{'PROGRAMFILES':'C:/Program Files'}), patch('local_commands.subprocess.Popen') as popen:
            self.assertTrue(router._open_chrome('https://www.youtube.com/'))
        args=popen.call_args.args[0]
        self.assertEqual(len(args),2)
        self.assertTrue(args[0].endswith('chrome.exe'))
        self.assertEqual(args[1],'https://www.youtube.com/')
        self.assertIs(popen.call_args.kwargs['shell'],False)

    def test_lookup_is_flat_metadata_only_and_validates_id(self):
        client=MagicMock()
        client.__enter__.return_value=client
        client.extract_info.return_value={'entries':[{'id':'BaW_jenozKc','title':'Test video'}]}
        # Avoid importing/installing a network library just to run offline tests.
        module=NS(YoutubeDL=MagicMock(return_value=client))
        with patch.dict('sys.modules',{'yt_dlp':module}):
            self.assertEqual(find_video('test video'),Video('BaW_jenozKc','Test video'))
            client.extract_info.assert_called_once_with('ytsearch1:test video',download=False)
            options=module.YoutubeDL.call_args.args[0]
            self.assertTrue(options['extract_flat'])
            self.assertTrue(options['skip_download'])
            self.assertEqual(options['retries'],0)
            self.assertFalse(options['cachedir'])
            client.extract_info.return_value={'entries':[{'id':'https://evil.invalid/'}]}
            with self.assertRaises(MediaLookupError): find_video('test video')

    def make_engine(self, confidence):
        output=NS(sentences=queue.Queue(),interrupt=lambda:None)
        engine=Inference('',Generation(),threading.Event(),queue.Queue(),output,
                         NS(reserve=lambda n:self.fail('Gemini quota must not be used')),
                         local_recognizer=NS(transcribe=lambda wav:('open youtube',confidence)),
                         local_router=LocalRouter(),local_only=False)
        return engine,output

    def test_low_confidence_media_does_not_fall_back_to_gemini(self):
        engine,output=self.make_engine(.6)
        with patch.object(engine.local_router,'execute') as execute:
            self.assertTrue(asyncio.run(engine.handle_local(Turn(0,b'',b'',0))))
        execute.assert_not_called()
        self.assertIn('repeat',output.sentences.get_nowait()[1])

    def test_confident_media_uses_local_route_without_quota(self):
        engine,output=self.make_engine(.99)
        with patch.object(engine.local_router,'_open_chrome',return_value=True) as launch:
            self.assertTrue(asyncio.run(engine.handle_local(Turn(0,b'',b'',0))))
        launch.assert_called_once()
        self.assertIn('YouTube',output.sentences.get_nowait()[1])


if __name__=='__main__': unittest.main()
