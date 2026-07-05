.PHONY: setup data popularity als twotower eval test

setup:
	pip install -r requirements.txt

data:
	python data_download.py

popularity:
	python -m src.popularity

als:
	python -m src.als

twotower:
	python -m src.two_tower

eval:
	python -m src.evaluate

test:
	pytest tests/ -v
