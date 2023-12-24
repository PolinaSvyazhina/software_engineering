FROM python:3.10

COPY requirements.txt requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8000:8000

COPY fastApiPractice.py fastApiPractice.py

CMD uvicorn fastApiPractice:app --host 0.0.0.0 --port 8000 --reload
