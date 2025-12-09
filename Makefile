.PHONY: install train predict app api test docker-build docker-run

install:
	@pip install -r requirements.txt

train:
	@python main.py train

predict:
	@python main.py predict

app:
	@python main.py streamlit

api:
	@python main.py api

test:
	@pytest || echo "No tests yet"

docker-build:
	@docker build -t glass-ensemble-ml .

docker-run:
	@docker run -p 8501:8501 glass-ensemble-ml
