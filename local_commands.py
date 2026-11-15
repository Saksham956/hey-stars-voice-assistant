"""Conservative local intents. No cloud calls, shell strings, or audio callbacks."""
from dataclasses import dataclass
from contextlib import closing
from concurrent.futures import Future, InvalidStateError
from datetime import datetime
import io
import json
import logging
import os
import queue
from pathlib import Path
import re
import sqlite3
import subprocess
import threading
import wave
from urllib.parse import urlencode

LOG=logging.getLogger('stars')

@dataclass(frozen=True)
class Intent:
    name: str
    query: str = ''

@dataclass(frozen=True)
class Result:
    message: str
    speak: bool = True

class LocalRouter:
    ALIASES = {
        'time': {'what time is it', 'what is the time', 'tell me the time', 'current time'},
        'chrome': {'open chrome', 'open google chrome', 'launch chrome'},
        'youtube': {'open youtube', 'open you tube', 'open youtube in chrome',
                    'open you tube in chrome', 'open youtube on chrome'},
        'volume_up': {'increase volume', 'turn up the volume', 'volume up'},
        'volume_down': {'decrease volume', 'turn down the volume', 'volume down'},
        'tasks': {'show my tasks', 'show my task', 'show tasks', 'what are my tasks',
                  'list tasks', 'list my tasks', 'task list', 'read my tasks'},
        'stop': {'stop speaking', 'stop talking', 'be quiet'},
    }
    def __init__(self, tasks_path=None):
        self.tasks_path=Path(tasks_path) if tasks_path else Path(__file__).with_name('local_tasks.sqlite3')
    def match(self, text):
        normalized=re.sub(r'[^a-z0-9\s]', '', text.lower())
        normalized=' '.join(normalized.split())
        for name,phrases in self.ALIASES.items():
            if normalized in phrases: return Intent(name)
        # Anchor the whole utterance. The title is data for a YouTube search,
        # never an executable, arbitrary URL or a second desktop command.
        utterance=' '.join(text.strip().split())
        play=re.fullmatch(
            r'(?:please\s+)?(?:open (?:youtube|you tube)(?: (?:in|on) chrome)? and )?'
            r'play\s+(?:(?:the\s+)?(?:song|video)\s+)?(.+?)'
            r'(?:\s+on\s+(?:youtube|you tube))?(?:\s+in\s+chrome)?[.!?]*',utterance,re.I)
        search=re.fullmatch(
            r'(?:please\s+)?(?:search (?:youtube|you tube) for\s+(.+?)|'
            r'(?:search for|find)\s+(.+?)\s+on\s+(?:youtube|you tube))'
            r'(?:\s+in\s+chrome)?[.!?]*',utterance,re.I)
        if play or search:
            match=play or search
            query=next(value for value in match.groups() if value is not None).strip(' .!?')
            if 1 <= len(query) <= 160 and any(c.isalnum() for c in query):
                return Intent('youtube_play' if play else 'youtube_search',query)
        return None
    def _open_chrome(self, url=None):
        roots=[os.environ.get('PROGRAMFILES',''),os.environ.get('PROGRAMFILES(X86)',''),os.environ.get('LOCALAPPDATA','')]
        for root in roots:
            if not root: continue
            exe=Path(root)/'Google/Chrome/Application/chrome.exe'
            if exe.is_file():
                subprocess.Popen([str(exe)]+([url] if url else []),shell=False)
                return True
        return False
    def execute(self, intent, *, is_current=lambda: True):
        if not is_current(): return Result('Command cancelled.',speak=False)
        if intent.name=='time':
            return Result('It is '+datetime.now().strftime('%I:%M %p').lstrip('0')+'.')
        if intent.name=='stop': return Result('Speech stopped.',speak=False)
        if intent.name=='tasks':
            if not self.tasks_path.exists(): return Result('You have no locally saved tasks yet. WhatsApp and calendar imports are not connected.')
            # Read-only: importing tasks belongs to the later academic workflow.
            with closing(sqlite3.connect(self.tasks_path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
                rows=db.execute("SELECT title FROM tasks WHERE status='pending' ORDER BY id LIMIT 5").fetchall()
            return Result('Your pending tasks: '+ '; '.join(r[0] for r in rows) if rows else 'No pending local tasks.')
        if intent.name=='chrome':
            if self._open_chrome(): return Result('Opening Chrome.')
            return Result('Chrome was not found in its standard Windows install folders.')
        if intent.name in ('youtube','youtube_search','youtube_play'):
            message='Opening YouTube.'
            url='https://www.youtube.com/'
            if intent.name!='youtube':
                if not intent.query or len(intent.query)>160:
                    return Result('Say play, followed by the song or video name.')
                url='https://www.youtube.com/results?'+urlencode({'search_query':intent.query})
                message='Opening YouTube search results.'
            if intent.name=='youtube_play':
                from youtube_media import find_video, MediaLookupError
                LOG.info('LOCAL RESULT: Looking up your YouTube video...')
                try:
                    video=find_video(intent.query)
                    url='https://www.youtube.com/watch?'+urlencode({'v':video.video_id,'autoplay':'1'})
                    LOG.info('YouTube match: %s',video.title)
                    message='Opening the first YouTube match.'
                except MediaLookupError:
                    LOG.warning('YouTube lookup unavailable; opening search results instead; Gemini requests=0')
                    message='Direct play is unavailable. Opening YouTube search results.'
            # A wake interruption during network lookup invalidates this action.
            if not is_current(): return Result('Command cancelled.',speak=False)
            if not self._open_chrome(url):
                return Result('Chrome was not found in its standard Windows install folders.')
            return Result(message)
        if intent.name in ('volume_up','volume_down'):
            if os.name!='nt': return Result('Volume control requires Windows.')
            from windows_volume import change_volume
            change=change_volume(intent.name,is_current=is_current)
            if change is None: return Result('Command cancelled.',speak=False)
            LOG.info('Windows volume intent=%s before=%.4f after=%.4f muted=%s at_limit=%s',
                     intent.name,change.before,change.after,change.muted,change.at_limit)
            if change.at_limit:
                return Result('Volume is already at maximum.' if intent.name=='volume_up' else 'Volume is already at minimum.')
            message='Volume '+('increased' if intent.name=='volume_up' else 'decreased')+' to '+str(round(change.after*100))+' percent.'
            if change.muted: message+=' Audio is still muted.'
            return Result(message)
        raise ValueError('Unsupported local intent')

class LocalRecognizer:
    def __init__(self, model_path):
        from vosk import Model, SetLogLevel
        SetLogLevel(-1)
        self.model=Model(str(model_path))
        self.lock=threading.Lock()
    def start_stream(self):
        return LocalSpeechStream(self)
    def transcribe(self, wav_bytes):
        from vosk import KaldiRecognizer
        with self.lock, wave.open(io.BytesIO(wav_bytes),'rb') as source:
            if (source.getnchannels(),source.getsampwidth(),source.getframerate())!=(1,2,16000):
                raise ValueError('Local recognition requires mono 16-bit 16kHz WAV')
            # Unrestricted recognition: a fixed grammar could force unrelated
            # questions into an allowed command and accidentally execute it.
            recognizer=KaldiRecognizer(self.model,16000)
            recognizer.SetWords(True)
            pieces=[]; confidences=[]
            def collect(value):
                pieces.append(value.get('text',''))
                confidences.extend(word.get('conf',0) for word in value.get('result',[]))
            while True:
                block=source.readframes(1600)
                if not block: break
                if recognizer.AcceptWaveform(block): collect(json.loads(recognizer.Result()))
            collect(json.loads(recognizer.FinalResult()))
            return ' '.join(p for p in pieces if p), min(confidences,default=0.0)

class LocalSpeechStream:
    """Decode one utterance on a worker, never on the microphone callback/worker.

    Only a final result can reach the command router. Bounded backpressure or
    cancellation invalidates the entire utterance rather than using partial text.
    """
    def __init__(self, owner):
        self.owner=owner
        self.blocks=queue.Queue(32)  # About one second of 512-sample frames.
        self.result=Future()
        self.cancelled=threading.Event()
        self.finished=False
        self.worker=threading.Thread(target=self.run,name='local-speech',daemon=True)
        self.worker.start()
    def cancel(self):
        self.cancelled.set()
        self.result.cancel()
    def _offer(self, block):
        if self.cancelled.is_set() or self.finished or self.result.done(): return
        try:
            self.blocks.put_nowait(block)
        except queue.Full:
            # Complete with a failure first so inference cannot fall back to cloud
            # on audio that may have lost a negation or another important word.
            try: self.result.set_exception(RuntimeError('Local speech processing queue full'))
            except InvalidStateError: pass
            self.cancelled.set()
    def feed(self, pcm):
        self._offer(pcm)
    def finish(self):
        self._offer(None)
        self.finished=True
        return self.result
    def run(self):
        acquired=False
        try:
            from vosk import KaldiRecognizer
            while not self.cancelled.is_set():
                acquired=self.owner.lock.acquire(timeout=.05)
                if acquired: break
            if self.cancelled.is_set(): return
            recognizer=KaldiRecognizer(self.owner.model,16000)
            recognizer.SetWords(True)
            pieces=[]; confidences=[]
            def collect(value):
                pieces.append(value.get('text',''))
                confidences.extend(word.get('conf',0) for word in value.get('result',[]))
            while not self.cancelled.is_set():
                try: block=self.blocks.get(timeout=.05)
                except queue.Empty: continue
                if block is None:
                    collect(json.loads(recognizer.FinalResult()))
                    if not self.result.done():
                        self.result.set_result((' '.join(p for p in pieces if p),min(confidences,default=0.0)))
                    return
                if recognizer.AcceptWaveform(block): collect(json.loads(recognizer.Result()))
        except Exception as exc:
            if not self.result.done(): self.result.set_exception(exc)
            self.cancelled.set()
        finally:
            if acquired: self.owner.lock.release()
