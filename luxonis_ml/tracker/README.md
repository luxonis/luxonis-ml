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

## Quickstart

`close()` finalizes every enabled backend. Use the tracker as a context
manager to call it automatically. If the block raises, the tracker marks the
run as failed:

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
finishes the WandB run. It is idempotent. If a backend fails to shut down, the
tracker reports the failure and does not raise it. Thus the teardown cannot
lose the data of the other backends.

## MLflow Notes

- MLflow is initialized lazily on first access to `tracker.experiment`.
- When MLflow logging fails, the tracker buffers the call. It replays the
  buffer in order when the server answers again. It retries a failed
  connection only once per minute, so an unreachable server does not stall the
  training loop. `close()` ignores that backoff and tries one last time.
- MLflow can reject one call and accept the calls behind it. The tracker then
  drops the rejected call, so one bad call cannot block every later log.
- The tracker copies a buffered artifact into the run directory immediately,
  because callers routinely delete the file after they hand it over.
- The buffer holds a maximum of 500 calls, of which 50 can be images. When the
  buffer is full, the tracker drops the oldest entry and warns once.
  Hyperparameters and artifacts go last, because no later call repeats them.
- On shutdown, any logs that could still not be sent are saved to:
  - `<save_directory>/<run_name>/local_logs.json`. A later save appends to
    this file and keeps the earlier records.
  - `<save_directory>/<run_name>/images/`
  - `<save_directory>/<run_name>/artifacts/<n>/`. Each artifact gets its own
    directory, which keeps artifacts with the same file name apart.

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
