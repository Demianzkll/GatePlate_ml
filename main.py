from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import os
from utils import PlateRecognizer

app = FastAPI()

recognizer = PlateRecognizer(
    model_path="weights/best.pt",
    tesseract_path=r"C:\Users\Demian\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)

@app.post("/api/recognize")
async def recognize(file: UploadFile = File(...)):
    tmp_path = None
    try:
        data = await file.read()

        fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        with os.fdopen(fd, "wb") as f:
            f.write(data)

        plate = recognizer.recognize_plate(tmp_path)

        return {"plate": plate, "found": plate != "Невпізнано"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
