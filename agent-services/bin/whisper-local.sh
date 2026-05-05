#!/bin/bash
# Local faster-whisper STT for OpenClaw tools.media.audio CLI model
# Usage: whisper-local.sh <audio_file_path>
exec /home/alansrobotlab/lloyd/agent-services/.venvs/whisper/bin/python -c "
import sys
from faster_whisper import WhisperModel
model = WhisperModel('small', device='cuda', compute_type='float16')
segments, _ = model.transcribe(sys.argv[1], beam_size=5, language='en', vad_filter=True)
print(' '.join(seg.text for seg in segments).strip())
" "$1"
