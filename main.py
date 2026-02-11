import os
import tempfile
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
TMP_SUFFIX = os.getenv("PYANNOTE_SUFFIX", ".wav")

if not HF_TOKEN:
    raise RuntimeError("Переменная окружения HF_TOKEN не задана (в .env или окружении).")

pipeline = Pipeline.from_pretrained(MODEL_ID, token=HF_TOKEN)

try:
    pipeline.to(DEVICE)
except Exception:
    pass


@app.post("/diarize")
async def diarize_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не передан.")

    with tempfile.NamedTemporaryFile(suffix=TMP_SUFFIX, delete=True) as tmp:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Пустой файл.")

        tmp.write(contents)
        tmp.flush()

        with ProgressHook() as hook:
            diarization = pipeline(tmp.name, hook=hook)

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
