"""Interactive A/B benchmark: one local question, two metered cloud requests.
Run from the STARS directory. Does not change the default model.
Only the first answer clause is spoken; full answers are displayed for review.
"""
import asyncio
from dataclasses import asdict
from datetime import datetime
import io
import json
from pathlib import Path
import threading
import time
import wave
import numpy as np
from hey_stars import (ScreenCache, RateGate, Clauses, HeardHeader, SYSTEM,
                       saved_api_key, SR)

async def compare(audio, jpeg, kokoro, sd, gate, client):
    from google.genai import types
    results=[]
    for model in ('gemini-2.5-flash', 'gemini-3.5-flash-lite'):
        # Shared counter with the main app; conservatively capped at 10/day.
        if not gate.reserve(8192):
            print('Local quota blocks this request. Stop here; do not reset the counter.')
            break
        print(f'\nTesting {model} (one request; no retries)...', flush=True)
        request_start=time.perf_counter()
        first_text=None
        full_text=''
        usage=None
        parser,header=Clauses(),HeardHeader()
        first_clause=None
        speech_future=None
        speech_error=[]
        speech_metrics={}
        loop=asyncio.get_running_loop()
        def speak(clause):
            try:
                synth_start=time.perf_counter()
                samples,rate=kokoro.create(clause,voice='af_sarah',speed=1.05,lang='en-us')
                samples=np.asarray(samples,dtype=np.float32).reshape(-1)
                speech_metrics['first_clause_tts_seconds']=time.perf_counter()-synth_start
                position=0
                finished=threading.Event()
                def playback(outdata,frames,timing,status):
                    nonlocal position
                    outdata.fill(0)
                    n=min(frames,len(samples)-position)
                    if n:
                        if position==0:
                            speech_metrics['request_to_first_pcm_seconds']=time.perf_counter()-request_start
                        outdata[:n,0]=samples[position:position+n]
                        position+=n
                    if position>=len(samples): finished.set()
                with sd.OutputStream(samplerate=rate,channels=1,dtype='float32',blocksize=480,latency='low',callback=playback):
                    if not finished.wait(20): raise TimeoutError('Playback timed out')
            except Exception as exc:
                speech_error.append(type(exc).__name__)
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM+' First output HEARD: and a faithful audio transcription on one line. Then a newline and the answer without a label.',
            max_output_tokens=256,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=(types.ThinkingConfig(thinking_budget=0) if model.startswith('gemini-2.') else types.ThinkingConfig(thinking_level='minimal')))
        stream=None
        error=None
        try:
            async with asyncio.timeout(20):
                stream=await client.aio.models.generate_content_stream(model=model,config=config,contents=[types.Content(role='user',parts=[
                    types.Part.from_text(text='Answer the spoken question using this screen.'),
                    types.Part.from_bytes(data=jpeg,mime_type='image/jpeg'),
                    types.Part.from_bytes(data=audio,mime_type='audio/wav')])])
                async for chunk in stream:
                    if chunk.usage_metadata:
                        usage=chunk.usage_metadata.model_dump(mode='json',exclude_none=True)
                    text=chunk.text or ''
                    if text and first_text is None: first_text=time.perf_counter()-request_start
                    full_text+=text
                    clauses=parser.feed(header.feed(text))
                    if clauses and speech_future is None:
                        first_clause=clauses[0]
                        speech_future=loop.run_in_executor(None,speak,first_clause)
                tail=parser.feed(header.finish())+parser.finish()
                if tail and speech_future is None:
                    first_clause=tail[0]
                    speech_future=loop.run_in_executor(None,speak,first_clause)
        except Exception as exc:
            error=type(exc).__name__
            if getattr(exc,'code',None)==429: gate.block()
        finally:
            if stream is not None: await stream.aclose()
        if speech_future is not None: await speech_future
        result=dict(model=model,first_text_seconds=first_text,**speech_metrics,
                    usage=usage,error=error,tts_errors=speech_error)
        results.append(result)
        print('ANSWER / TRANSCRIPT:\n'+full_text)
        print('METRICS:\n'+json.dumps(result,indent=2))
        if error:
            print('Stopping after the error to avoid another cloud request.')
            break
        await asyncio.sleep(1)
    # Do not persist screen, recorded speech, transcript, key, or answer text.
    output=Path(__file__).with_name('benchmark_results.jsonl')
    with output.open('a',encoding='utf-8') as file:
        for result in results:
            file.write(json.dumps(dict(timestamp=datetime.now().isoformat(),**result))+'\n')
    print('\nSaved numeric metrics to benchmark_results.jsonl.')
    print('TTFA here starts at REQUEST SUBMISSION, excluding VAD/endpointing and hardware latency.')
    print('Do not compare that directly to the main app end-of-speech TTFA. One pair is not a reliable average.')

def main():
    import sounddevice as sd
    from kokoro_onnx import Kokoro
    from google import genai
    from google.genai import types
    key=saved_api_key()
    if not key: raise SystemExit('Saved key not found. No requests made.')
    stop=threading.Event()
    screen=ScreenCache(stop)
    worker=threading.Thread(target=screen.run,daemon=True)
    worker.start()
    try:
        print('Warming local Kokoro; no API call...')
        k=Kokoro('models/kokoro-v1.0.onnx','models/voices-v1.0.bin')
        k.create('Ready.',voice='af_sarah',speed=1.05,lang='en-us')
        input('Close the normal STARS app. Press Enter, then ask ONE short screen question within 4 seconds. Two API requests will be used: ')
        print('Recording now...',flush=True)
        samples=sd.rec(4*SR,samplerate=SR,channels=1,dtype='int16')
        sd.wait()
        if np.max(np.abs(samples.astype(np.int32)))<100:
            raise SystemExit('Recording was nearly silent; no requests made.')
        jpeg=screen.snapshot()
        wav=io.BytesIO()
        with wave.open(wav,'wb') as writer:
            writer.setnchannels(1);writer.setsampwidth(2);writer.setframerate(SR)
            writer.writeframes(samples.astype('<i2').tobytes())
        gate=RateGate(Path(__file__).with_name('hey_stars_usage.sqlite3'),5,250000,10)
        client=genai.Client(api_key=key,http_options=types.HttpOptions(timeout=15000,retry_options=types.HttpRetryOptions(attempts=1)))
        async def run():
            try: await compare(wav.getvalue(),jpeg,k,sd,gate,client)
            finally: await client.aio.aclose()
        asyncio.run(run())
    finally: stop.set()

if __name__=='__main__': main()
