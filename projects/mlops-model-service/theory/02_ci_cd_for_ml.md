# CI/CD for ML

## Why ML CI/CD differs from plain software CI/CD

Standard CI tests code. ML CI must also test:

- **The model.** Does it meet a minimum accuracy threshold?
- **The data contract.** Does the input schema hold?
- **The serving path.** Does the new model produce the same output format?
- **Backward compatibility.** Do existing downstream integrations still work?

Skipping these checks in ML CI leads to deploys that are green on code-level tests and silently degrade production quality.

## The pipeline in this project

GitHub Actions workflow in `.github/workflows/mlops-model-service.yml`. Stages:

### 1. Lint
- `ruff check app/ tests/`
- Fast (seconds).
- Blocks the rest on any lint error.

### 2. Tests
- `pytest tests/`
- Includes schema validation and API tests with a mocked predictor.
- Runs in ~10 to 30 seconds.
- Model tests (with real weights) are a separate, optional stage (slow, requires GCS download).

### 3. Build
- `docker build -t gcr.io/$PROJECT/nlp-reviews:$SHA .`
- Multi-stage build keeps the final image small.

### 4. Push to registry
- Only on `main` branch.
- Authenticates via `GCP_SA_KEY` GitHub secret.

### 5. Deploy to Cloud Run
- Only on `main` branch.
- `gcloud run deploy` to a staging service first (different service name).
- Smoke test the staging URL.
- Promote traffic to production via `gcloud run services update-traffic`.

### 6. Rollback
- Not automatic. If staging smoke tests fail, the pipeline fails and production is untouched.
- Manual rollback: `gcloud run services update-traffic --to-revisions=PREVIOUS=100`.

## The CI/CD configuration

```yaml
# .github/workflows/mlops-model-service.yml (sketch)
name: mlops-model-service
on:
  push:
    branches: [main]
    paths:
      - "projects/mlops-model-service/**"
      - ".github/workflows/mlops-model-service.yml"
  pull_request:
    paths:
      - "projects/mlops-model-service/**"

jobs:
  test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: projects/mlops-model-service
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: ruff check app/ tests/
      - run: pytest tests/ -v

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write
    steps:
      - uses: actions/checkout@v4
      - uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      - uses: google-github-actions/setup-gcloud@v2
      - name: Build and deploy
        env:
          PROJECT: ${{ secrets.GCP_PROJECT }}
          SERVICE: nlp-reviews
          REGION: us-central1
        run: |
          cd projects/mlops-model-service
          bash scripts/deploy_cloud_run.sh
```

## Secrets management

Two secrets live in GitHub Actions:

- `GCP_SA_KEY`: service account JSON with Cloud Run Admin, Storage Admin, and Service Account User roles scoped to one project.
- `GCP_PROJECT`: GCP project ID.

Never committed to code. Accessed via `${{ secrets.NAME }}` in the workflow.

For higher security: Workload Identity Federation (WIF) instead of long-lived keys. WIF exchanges GitHub's OIDC token for a short-lived GCP token. Slightly more setup, but eliminates the long-lived key liability. Documented but not implemented here.

## Validating the model in CI

This service does not retrain the model. But a more complete pipeline would include a model-validation stage:

1. Load the candidate model artifact (from GCS or MLflow).
2. Evaluate on a held-out "canary" dataset.
3. Compare metrics (macro F1) to the current production model.
4. If worse than production by more than a threshold, fail the pipeline.
5. If better or equal, proceed to deploy.

Tools: MLflow's model registry with `validate` transitions, or a custom GitHub Action that calls the evaluation script.

## Blue-green vs canary rollout

- **Blue-green.** Deploy new version alongside old. Switch 100 percent of traffic at once. Fast rollback.
- **Canary.** Send a small percent to new, watch metrics, increase. Slower rollout, safer.
- **Cloud Run traffic split.** Supports weighted routing across revisions natively. Canary is one `gcloud run services update-traffic --to-revisions NEW=10,OLD=90` command.

This project uses blue-green (0 or 100) by default for simplicity. Canary is documented as an upgrade path.

## Monitoring as part of CD

Post-deploy, the CI should:

1. Hit `/health` on the new URL.
2. Send 1 to 5 sample requests; assert correct response format.
3. If any fail, roll back automatically.

The `scripts/deploy_cloud_run.sh` ends with a smoke test. In CI, this guard-rails the deploy.

## Why the CI badge on the main README matters

A green CI badge is one of the first things a recruiter looks at. It signals:

- Tests exist.
- Tests run on every commit.
- The codebase is not broken right now.

Badge markdown:

```markdown
[![mlops-model-service CI](https://github.com/GonorAndres/data-science-path/actions/workflows/mlops-model-service.yml/badge.svg)](https://github.com/GonorAndres/data-science-path/actions/workflows/mlops-model-service.yml)
```

The badge sits at the top of the root README with the tagline.

## Cross-links

- `01_serving_patterns.md` for the service architecture.
- `03_monitoring_and_drift.md` for post-deploy observability.
- `docs/theory/13_mlops_architecture_patterns.md` for the broader MLOps picture.
- `docs/theory/14_cloud_deployment_primer.md` for cloud specifics.
