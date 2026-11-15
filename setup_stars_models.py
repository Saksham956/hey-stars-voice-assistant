"""Download public inference assets; does not use a Gemini key or API."""
from pathlib import Path
from urllib.request import urlopen
import shutil

ASSETS = {
    'melspectrogram.onnx': 'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx',
    'embedding_model.onnx': 'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx',
    'kokoro-v1.0.onnx': 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx',
    'voices-v1.0.bin': 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin',
}

def main():
    folder = Path(__file__).with_name('models')
    folder.mkdir(exist_ok=True)
    for name, url in ASSETS.items():
        target = folder / name
        if target.exists():
            print(f'Already present: {name}', flush=True)
            continue
        temp = target.with_suffix(target.suffix + '.part')
        print(f'Downloading {name}...', flush=True)
        try:
            with urlopen(url, timeout=10) as response, temp.open('wb') as output:
                shutil.copyfileobj(response, output)
            temp.replace(target)
        except Exception as exc:
            temp.unlink(missing_ok=True)
            raise SystemExit(f'Download failed ({type(exc).__name__}). Run this script again from your terminal.')
    print('Support assets ready. A trained custom models/hey_stars.onnx is still required.')

if __name__ == '__main__': main()
