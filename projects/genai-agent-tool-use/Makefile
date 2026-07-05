.PHONY: setup agent eval test

setup:
	pip install -r requirements.txt

agent:
	python -m src.agent "$(TASK)"

eval:
	python -m src.evaluate

test:
	pytest tests/ -v
