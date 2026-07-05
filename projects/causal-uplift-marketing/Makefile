.PHONY: setup data t-learner s-learner r-learner eval ab-sim test

setup:
	pip install -r requirements.txt

data:
	python data_download.py

t-learner:
	python -m src.t_learner

s-learner:
	python -m src.s_learner

r-learner:
	python -m src.r_learner

eval:
	python -m src.evaluate

ab-sim:
	python -m src.ab_simulation

test:
	pytest tests/ -v
