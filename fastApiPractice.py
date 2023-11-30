from transformers import pipeline
from PIL import Image
from fastapi import FastAPI, File, UploadFile

app = FastAPI()
cls = pipeline("image-classification", model="gianlab/swin-tiny-patch4-window7-224-finetuned-ecg-classification")


@app.get('/')
async def root():
    return {'message': 'Hello Word, this is image-classification model'}


@app.post('/predict/')
async def image_classification(file: UploadFile = File(...)):
    img = Image.open(file.file)
    img = img.resize((224, 224))
    result = cls(img)
    return max(result, key=lambda x: x['score'])['label']
