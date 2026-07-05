.PHONY: setup data baseline train eval demo test clean

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

data:
	python data_download.py

baseline:
	python -m src.baseline

train:
	python -m src.train

eval:
	python -m src.evaluate

demo:
	streamlit run .streamlit/app.py

test:
	pytest tests/ -v

clean:
	rm -rf data/ models/ mlruns/
