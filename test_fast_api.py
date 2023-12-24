from fastapi.testclient import TestClient
from fastApiPractice import app

client = TestClient(app)


def test_read_main():
    responses = client.get("/")
    assert responses.status_code == 200
    assert responses.json() == {'message': 'Hello Word, this is image-classification model'}


def test_predict_empty():
    responses = client.post('/predict/')
    assert responses.status_code == 422
