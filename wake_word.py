"""Optional offline wake phrase listener. No cloud calls."""
import json
import threading
from pathlib import Path
import sounddevice as sd

MODEL_PATH = Path(__file__).with_name('vosk-model-small-en-us-0.15')

class WakeListener:
    def __init__(self, notify):
        self.notify = notify
        self.stop_event = threading.Event()
        self.thread = None
        self.model = None

    def stop(self):
        self.stop_event.set()

    def start(self):
        if self.thread and self.thread.is_alive():
            return False
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.listen, daemon=True)
        self.thread.start()
        return True

    def listen(self):
        try:
            from vosk import Model, KaldiRecognizer
            if not MODEL_PATH.is_dir():
                raise RuntimeError('Download and extract vosk-model-small-en-us-0.15 into the STARS folder first.')
            if self.model is None:
                self.model = Model(str(MODEL_PATH))
            recognizer = KaldiRecognizer(self.model, 16000, json.dumps(['hey stars', '[unk]']))
            detected = False
            with sd.RawInputStream(samplerate=16000, blocksize=1600, channels=1, dtype='int16') as mic:
                self.notify('wake_status', 'Local listening: say Hey STARS. Pause after the phrase.')
                while not self.stop_event.is_set():
                    data, overflow = mic.read(1600)
                    final = recognizer.AcceptWaveform(bytes(data))
                    result = json.loads(recognizer.Result() if final else recognizer.PartialResult())
                    if result.get('text', result.get('partial', '')) == 'hey stars':
                        detected = True
                        break
            # Release the microphone BEFORE the main assistant starts recording.
            if detected and not self.stop_event.is_set():
                self.notify('wake', '')
        except Exception as exc:
            self.notify('wake_error', f'Local wake word unavailable: {exc}')
