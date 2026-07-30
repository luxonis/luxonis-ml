# LuxonisML Tracker

## Introduction

LuxonisML Tracker provides a unified logging interface for common experiment
tracking backends. It wraps TensorBoard, Weights & Biases, and MLflow behind a
single class so training pipelines can log metrics, images, artifacts, and
matrices without backend-specific code.

The main entry point is `LuxonisTracker`.

## Table of Contents

- [LuxonisML Tracker](#luxonisml-tracker)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Core Components](#core-components)
  - [Quickstart](#quickstart)
  - [Backend Examples](#backend-examples)
  - [Logging API](#logging-api)
  - [MLflow Notes](#mlflow-notes)
  - [Distributed Training](#distributed-training)

## Installation

Install tracker extras and the backend you want to use:

```bash
pip install luxonis-ml[tracker]
```

Additional backend dependencies:

- TensorBoard: `torch` (for `torch.utils.tensorboard`)
- Weights & Biases: `wandb`
- MLflow: `luxonis-ml[mlflow]` or `mlflow`

> [!NOTE]
> The tracker does not force-install backend SDKs. Make sure the backend you
> want is available in your environment.

## Core Components

- **LuxonisTracker**

  - The unified logging facade. You decide which backends to enable via
    constructor flags (`is_tensorboard`, `is_wandb`, `is_mlflow`).

- **LuxonisRequestHeaderProvider**

  - MLflow request header provider that injects Cloudflare access headers from
    environment variables when needed. It requires MLflow, so it is imported
    lazily on first access.

## Quickstart

`close()` finalizes every enabled backend, so use the tracker as a context
manager and it is called for you — with the run marked as failed if the block
raises:

```python
from luxonis_ml.tracker import LuxonisTracker

with LuxonisTracker(
    project_name="my-project",
    run_name="baseline",
    is_tensorboard=True,
) as tracker:
    for step in range(10):
        tracker.log_metric("loss", 0.1 * (10 - step), step)
```

## Backend Examples

### TensorBoard

```python
from luxonis_ml.tracker import LuxonisTracker

tracker = LuxonisTracker(
    project_name="my-project",
    run_name="tb-run",
    is_tensorboard=True,
)

tracker.log_metrics({"acc": 0.92, "loss": 0.18}, step=1)
tracker.close()
```

### Weights & Biases

```python
from luxonis_ml.tracker import LuxonisTracker

tracker = LuxonisTracker(
    project_name="my-project",
    run_name="wandb-run",
    is_wandb=True,
    wandb_entity="my-entity",
)

tracker.log_metric("loss", 0.42, step=1)
tracker.upload_artifact("model.onnx", name="model", typ="model")
tracker.close()
```

### MLflow

```python
from luxonis_ml.tracker import LuxonisTracker

tracker = LuxonisTracker(
    project_name="my-project",
    run_name="mlflow-run",
    is_mlflow=True,
    mlflow_tracking_uri="http://localhost:5000",
)

tracker.log_hyperparams({"lr": 1e-3, "batch_size": 32})
tracker.log_metric("mAP", 0.55, step=1)
tracker.close()
```

## Logging API

Common methods exposed by `LuxonisTracker`:

- `log_hyperparams(params)`
- `log_metric(name, value, step)`
- `log_metrics(metrics, step)`
- `log_image(name, img, step)`
- `log_images(imgs, step)`
- `upload_artifact(path, name=None, typ="artifact")`
- `log_matrix(matrix, name, step)`
- `close(status="success")`

Images are expected as `numpy.ndarray` and are logged in a backend-appropriate
format.

`close()` flushes and closes the TensorBoard writer, ends the MLflow run, and
finishes the WandB run. It is idempotent, and a backend that fails to shut down
is reported rather than raised, so teardown cannot lose the other backends'
data.

## MLflow Notes

- MLflow is initialized lazily on first access to `tracker.experiment`.
- When MLflow logging fails, the call is buffered and replayed — in order —
  once the server answers again. A failed connection is only retried once per
  minute, so an unreachable server does not stall the training loop.
- The buffer is bounded (500 calls, of which at most 50 images). The oldest
  entries are dropped, with a warning, once it fills up.
- On shutdown, any logs that could still not be sent are saved to:
  - `<save_directory>/<run_name>/local_logs.json`
  - `<save_directory>/<run_name>/images/`
  - `<save_directory>/<run_name>/artifacts/`

### Cloudflare Access Headers

If your MLflow endpoint is protected by Cloudflare Access, set:

- `MLFLOW_CLOUDFLARE_ID`
- `MLFLOW_CLOUDFLARE_SECRET`

The `LuxonisRequestHeaderProvider` injects those headers into MLflow requests.

## Distributed Training

Pass `rank` to only log from the primary process:

```python
tracker = LuxonisTracker(
    project_name="my-project",
    is_tensorboard=True,
    rank=0,
)
```

All logging methods are decorated with a rank check, so only rank 0 writes.
