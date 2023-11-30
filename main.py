import io
import streamlit as st
from transformers import pipeline
from PIL import Image

st.header("Классификация ЭКГ")


def get_label_description(label):
    match label:
        case 'N':
            st.title("Нормальный ритм")
        case "S":
            st.title("Суправентрикулярная экстрасистолия")
        case "V":
            st.title("Преждевременное сокращение желудочков")
        case "F":
            st.title("Слияние желудочкового и нормального сокращений")
        case "?":
            st.title("Неклассифицируемый бит")
        case "M":
            st.title("Инфаркт миокарда")


def load_image():
    uploaded_file = st.file_uploader(
        label='Добавьте изображение'
    )
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


img = load_image()
result = []
cls = pipeline("image-classification", model="gianlab/swin-tiny-patch4-window7-224-finetuned-ecg-classification")
if img is not None:
    img = img.resize((224, 224))
    result = cls(img)

if result is not None:
    get_label_description(max(result, key=lambda x: x['score'])['label'])
