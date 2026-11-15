# STARS — Windows Hey Jarvis voice assistant

A Windows Python voice assistant with local wake detection, speech recognition,
spoken replies, a desktop status panel, and an optional Gemini fallback for
questions about the screen. The current launcher uses the **Hey Jarvis** wake
model; the project name does not change the phrase a model recognizes.

## Implemented features

- openWakeWord wake detection, Silero voice activity detection, Vosk streaming
  recognition, and Kokoro ONNX speech synthesis.
- Exact local commands for time, Chrome, volume, saved tasks, and YouTube.
- Wake interruption cancels stale replies and pending actions.
- Optional Gemini audio/image requests with a persistent local quota gate.
- A Tk status/transcript panel and an older Gemini Live GUI in `stars.py`.

This is an existing personal project, not a production reliability claim.
Hardware behavior, recognition, provider model availability, and browser playback
need validation on each machine. No latency guarantee is made.

## Windows setup

Use a separate virtual environment. Python 3.11 is a reasonable starting point
if local ML packages do not offer wheels for a newer Python version.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-low-latency.txt
.\.venv\Scripts\python.exe setup_stars_models.py
```

The existing download helper obtains four support/TTS assets. It does **not**
download the wake classifier or Vosk speech model. Obtain those separately:

- Place the official pretrained `hey_jarvis_v0.1.onnx` classifier from
  [openWakeWord](https://github.com/dscripka/openWakeWord) in `models/`.
  Speak its trained phrase, "Hey Jarvis". For "Hey Stars", train a genuine
  custom classifier and pass its path explicitly; do not rename a Jarvis model.
- Download and extract `vosk-model-small-en-in-0.4` from the
  [Vosk model catalog](https://alphacephei.com/vosk/models) to
  `models/vosk-model-small-en-in-0.4/`.
- The helper supplies `melspectrogram.onnx`, `embedding_model.onnx`,
  `kokoro-v1.0.onnx`, and `voices-v1.0.bin` under `models/`.

Respect each dependency and model's license. Model files are excluded from Git.
Downloads and ML runtimes can be large. The helper's final message refers to a
custom Hey Stars model; the included launcher instead selects Hey Jarvis.

## Run local commands first

List devices, then replace `1` with your input microphone's device number:

```powershell
.\.venv\Scripts\python.exe -m sounddevice
powershell -File .\start_fast_stars.ps1 -LocalOnly -InputDevice 1
```

The launcher retains the original machine's microphone-name default. Always
provide your own device number or `-InputDeviceName` on another machine.
Say "Hey Jarvis", wait for the beep, then try "What time is it?", "Open Chrome",
"Increase volume", "Show my tasks", or "Search YouTube for Python tutorial".
Task reading uses a local `local_tasks.sqlite3` database if present; this project
does not import tasks from messaging or calendar accounts.

`-LocalOnly` prevents Gemini requests. YouTube commands still use the internet
and open Chrome; play-by-title uses yt-dlp metadata lookup. No media download or
browser-cookie import is implemented. Autoplay and search results can vary.

## Optional Gemini mode and privacy

Set `GEMINI_API_KEY` in your Windows user environment using your own secure
configuration workflow. The backend reads the process environment or the saved
Windows user variable. It does not automatically load `.env` files. Never commit
a key or put it in a command-line argument.

Normal mode captures microphone commands and a screen image for Gemini fallback.
The backend maintains a screen preview in memory while running. Local matched
commands bypass Gemini; unmatched questions in normal mode can send audio and
screen contents to Google. Close confidential windows before using cloud mode.
Transcripts/replies can appear in the panel and console; redirected logs are
private data. Usage databases and personal configuration remain local.

The existing launcher hardcodes RPM 5, TPM 250000, RPD 10. These are local
configuration values, not verified current provider limits. Before cloud use,
check your own project's limits and use explicit values, for example:

```powershell
$starsRpm = [int](Read-Host 'Your RPM limit or lower')
$starsTpm = [int](Read-Host 'Your input TPM limit or lower')
$starsRpd = [int](Read-Host 'Your daily limit or lower')
.\.venv\Scripts\python.exe hey_stars.py --ui --wake-model models/hey_jarvis_v0.1.onnx --local-stt-model models/vosk-model-small-en-in-0.4 --device 1 --rpm $starsRpm --tpm $starsTpm --rpd $starsRpd
```

The local daily cap is at most 10 requests. It cannot account for other apps'
usage and cannot guarantee provider billing limits. Provider model names in
the original code may require updates as availability changes.

## Offline tests and utilities

```powershell
.\.venv\Scripts\python.exe run_offline_tests.py --timeout 60
```

The four included test modules mock cloud/audio/model/browser interactions.
They do not establish live headset or cloud performance. Results are written to
the ignored `verification/` folder. `test_connection.py` is a separate manual
network check that prompts for a key. `benchmark_stars.py` can make real cloud
requests; review its inputs before using it. `verify_stars_volume.ps1` reads the
Windows output endpoint; its `-TestActions` option changes volume temporarily.

Initial import validation: all 61 existing offline tests passed on Windows on
November 16, 2026. Source files parsed successfully, and the selected upload files
were checked for common credential patterns, private keys, email addresses,
personal paths, and large files. No matches were found. This is a bounded scan,
not a guarantee that every possible security issue has been identified.

For the older GUI, install `requirements.txt`, extract the Vosk US-English model
at the repository root, then run `stars.py` without `--low-latency`. This legacy
path and its configured Gemini Live model have not been validated for release.

## Repository contents and rights

Source and existing tests were copied from the working Windows project.
Virtual environments, downloaded models, databases, runtime usage, logs,
recordings, screenshots, verification bundles and old backup copies are omitted.
The original local installation is preserved separately.

No open-source license is assigned in this initial private import. A license
should be selected by the owner after reviewing authorship and third-party terms.
