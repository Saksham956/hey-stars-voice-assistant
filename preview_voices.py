"""Audition installed Kokoro voices locally; no API key or cloud requests."""
import json
from pathlib import Path
import sounddevice as sd
from kokoro_onnx import Kokoro

OPTIONS = [('af_heart', 'American female'), ('am_michael', 'American male'),
           ('bf_emma', 'British female'), ('bm_george', 'British male')]

def main():
    root=Path(__file__).parent
    print('Loading local speech model. No Gemini requests will be made.')
    kokoro=Kokoro(str(root/'models/kokoro-v1.0.onnx'),str(root/'models/voices-v1.0.bin'))
    sample="Hello Saksham. I'm Stars, your desktop assistant. I can explain your screen and help you plan your next step."
    for number,(voice,label) in enumerate(OPTIONS,1):
        print(f'{number}: {voice} - {label}')
    while True:
        choice=input('Enter 1-4 to listen, S to save the last voice, or Q to quit: ').strip().lower()
        if choice=='q': break
        if choice=='s':
            if 'selected' not in locals():
                print('Listen to a voice first.'); continue
            (root/'voice_settings.json').write_text(json.dumps({'voice':selected,'speed':1.0},indent=2))
            print(f'Saved {selected}. Restart STARS to use it.'); break
        if choice not in ('1','2','3','4'): continue
        selected=OPTIONS[int(choice)-1][0]
        samples,rate=kokoro.create(sample,voice=selected,speed=1.0,
                                  lang='en-gb' if selected.startswith('b') else 'en-us')
        sd.play(samples,rate); sd.wait()

if __name__=='__main__':
    try: main()
    except KeyboardInterrupt: sd.stop()
