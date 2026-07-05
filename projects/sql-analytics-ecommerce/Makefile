.PHONY: setup data pipeline tests clean

setup:
	pip install -r requirements.txt

data:
	python data_download.py

pipeline:
	python -m src.pipeline

tests:
	python -m src.tests

pytest:
	pytest tests/ -v

clean:
	rm -rf data/*.duckdb
