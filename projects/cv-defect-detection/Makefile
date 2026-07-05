.PHONY: setup data train eval gradcam demo test

setup:
	pip install -r requirements.txt

data:
	python data_download.py

train:
	python -m src.train --category $(CATEGORY)

eval:
	python -m src.evaluate --category $(CATEGORY)

gradcam:
	python -m src.gradcam --category $(CATEGORY) --sample 0

demo:
	python app.py

test:
	pytest tests/ -v
