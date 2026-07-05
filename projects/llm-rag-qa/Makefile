.PHONY: setup data chunk embed eval demo test

setup:
	pip install -r requirements.txt

data:
	python data_download.py

chunk:
	python -m src.chunk

embed:
	python -m src.embed

eval:
	python -m src.evaluate

demo:
	python app.py

test:
	pytest tests/ -v
