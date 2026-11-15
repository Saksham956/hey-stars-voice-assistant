param([ValidateSet("gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.5-flash-lite")][string]$Model = "gemini-2.5-flash", [switch]$LocalOnly, [string]$SpeechModel = "models/vosk-model-small-en-in-0.4", [int]$InputDevice = -1, [string]$InputDeviceName = "Microphone (C-Media(R) Audio)", [ValidateRange(1,8)][int]$TtsThreads = 4, [switch]$BatchLocalSTT, [ValidateSet(1024,1280)][int]$ScreenMaxDimension = 1280)
# Quotas verified from your AI Studio screenshot. The engine additionally caps RPD at 10.
Set-Location -LiteralPath $PSScriptRoot
$starsExtraArgs = @('--tts-threads', $TtsThreads, '--screen-max-dimension', $ScreenMaxDimension)
if ($LocalOnly) { $starsExtraArgs += '--local-only' }
if ($BatchLocalSTT) { $starsExtraArgs += '--batch-local-stt' }
if ($InputDevice -ge 0) { $starsExtraArgs += @('--device', $InputDevice) }
else { $starsExtraArgs += @('--input-device-name', $InputDeviceName) }
$starsExtraArgs += @('--local-stt-model', (Join-Path $PSScriptRoot $SpeechModel))
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\stars.py" --low-latency --ui --wake-model "$PSScriptRoot\models\hey_jarvis_v0.1.onnx" --mel-model "$PSScriptRoot\models\melspectrogram.onnx" --embedding-model "$PSScriptRoot\models\embedding_model.onnx" --kokoro-model "$PSScriptRoot\models\kokoro-v1.0.onnx" --voices "$PSScriptRoot\models\voices-v1.0.bin" --rpm 5 --tpm 250000 --rpd 10 --model $Model @starsExtraArgs
