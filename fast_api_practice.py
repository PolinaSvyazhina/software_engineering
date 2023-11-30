from transformers import pipeline
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
import io

app = FastAPI()
cls = pipeline("image-classification", model="gianlab/swin-tiny-patch4-window7-224-finetuned-ecg-classification")


class Item(BaseModel):
    img: bytes


@app.get('/')
async def root():
    return {'message': 'Hello Word, this is image-classification model'}


@app.post('/img/')
async def image_create(img: Item):
    img = Image.open(io.BytesIO(img))
    return img.resize((224, 224))


@app.get('/redict/')
def predict(img: Item):
    img = Image.open(io.BytesIO(img))
    img = img.resize((224, 224))
    result = cls(img)
    return max(result, key=lambda x: x['score'])['label']
