.PHONY: setup lint test run docker-build docker-run copy-model deploy clean

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

lint:
	ruff check app/ tests/

test:
	pytest tests/ -v

run:
	uvicorn app.main:app --reload --port 8080

docker-build:
	docker build -t nlp-reviews-service:latest .

docker-run: docker-build
	docker run --rm -p 8080:8080 nlp-reviews-service:latest

copy-model:
	mkdir -p models
	cp -r ../nlp-reviews-multilingual/models/transformer/best models/

deploy:
	bash scripts/deploy_cloud_run.sh

clean:
	rm -rf .venv __pycache__ .pytest_cache .ruff_cache
