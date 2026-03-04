from fastapi import FastAPI, UploadFile, File
import shutil
from search import search_defect

app = FastAPI()   # VERY IMPORTANT - do not disable docs here

@app.get("/")
def root():
    return {"message": "Steel Defect AI Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = search_defect(temp_path)

    return {"prediction": results}