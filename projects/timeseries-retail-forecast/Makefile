.PHONY: setup data baselines classical gbm reconcile eval test clean

setup:
	pip install -r requirements.txt

data:
	python data_download.py

baselines:
	python -m src.baselines

classical:
	python -m src.classical

gbm:
	python -m src.gbm

reconcile:
	python -m src.reconcile

eval:
	python -m src.evaluate

test:
	pytest tests/ -v

clean:
	rm -rf data/ models/ mlruns/
