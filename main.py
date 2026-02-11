import os
import tempfile
import subprocess
from fastapi import FastAPI, UploadFile, File, HTTPException
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

MODEL_ID = os.getenv("PYANNOTE_MODEL_ID", "pyannote/speaker-diarization-community-1")
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = os.getenv("PYANNOTE_DEVICE", "cpu")
MERGE_GAP = float(os.getenv("PYANNOTE_MERGE_GAP", "0.5"))

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN не задан в .env или окружении.")

pipeline = Pipeline.from_pretrained(MODEL_ID, token=HF_TOKEN)
try:
    pipeline.to(DEVICE)
except Exception:
    pass


def ffmpeg_to_wav_48k_mono(src_path: str, dst_path: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-ac", "1",
        "-ar", "48000",
        "-c:a", "pcm_s16le",
        dst_path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr[-2000:]}")

@app.post("/diarize")
async def diarize_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не передан.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Пустой файл.")

    with tempfile.NamedTemporaryFile(suffix=".input", delete=True) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:

        tmp_in.write(contents)
        tmp_in.flush()

        try:
            ffmpeg_to_wav_48k_mono(tmp_in.name, tmp_wav.name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Не удалось декодировать аудио: {e}")

        with ProgressHook() as hook:
            diarization = pipeline(tmp_wav.name, hook=hook)

        merged_result = []
        previous = None

        for turn, speaker in diarization.speaker_diarization:
            start = float(turn.start)
            end = float(turn.end)

            if previous is None:
                previous = {"start": start, "end": end, "speaker": speaker}
                continue

            if speaker == previous["speaker"] and abs(start - previous["end"]) < MERGE_GAP:
                previous["end"] = end
            else:
                merged_result.append(previous)
                previous = {"start": start, "end": end, "speaker": speaker}

        if previous:
            merged_result.append(previous)

    return {"diarization": merged_result, "model": MODEL_ID, "device": DEVICE}
