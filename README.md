# https://huggingface.co/gianlab/swin-tiny-patch4-window7-224-finetuned-ecg-classification #

Модель работает с изображением ЭКГ размером 224px, 224px и имеет возможность классифицировать данные по следующим типам:

## Классифицированные данные: ##

* N: Нормальный ритм
* S: Суправентрикулярная экстрасистолия.
* V: Преждевременное сокращение желудочков.
* F: Слияние желудочкового и нормального сокращений.
* Вопрос: Неклассифицируемый бит
* М: инфаркт миокарда

## Запуск приложения и работа с приложением ##

1. В терменале, в нужной директории,  ввести команду: streamlit run main.py
2. Загрузить фото первого пика из ЭКГ
3. Получить результат