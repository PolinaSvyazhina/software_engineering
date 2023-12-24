from transformers import pipeline
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Response, status

app = FastAPI()
cls = pipeline("image-classification", model="gianlab/swin-tiny-patch4-window7-224-finetuned-ecg-classification")

# Рест для получения приветственного сообщения
# Может использоваться как показатель isHealth для мониторинга
@app.get('/')
async def root():
    return {'message': 'Hello Word, this is image-classification model'}


# Получить класс переданного изображения ЭКГ
@app.post('/predict/')
async def image_classification(response: Response, file: UploadFile = File(...)):
    if file.size == 0:
        response.status_code == status.HTTP_204_NO_CONTENT

        return

    img = Image.open(file.file)
    img = img.resize((224, 224))
    result = cls(img)
    return max(result, key=lambda x: x['score'])['label']
