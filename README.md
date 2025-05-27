# Modal demo usecases

## Setup

* Create a free account at modal.com
* Pip install modal


## Warmup
The `uv run` prefix is b/c we use the uv package manager.

* `uv run modal run hello_world.py` -- basics
* `uv run modal run gpu.py` -- how to provision GPUs.
* `uv run modal run fanout.py` -- basic fanout
* `uv run modal run web_api.py` -- hosting an API

## Training the DSVT backbone on modal.
This uses 4 * 8 A100 GPUs.
* https://github.com/beijbom/DSVT
* `uv run modal run --detach tools/scripts/modal_train.py --gpu a100 --n-gpus 4`

## Massive data processing jobs
* `uv run modal run --detach feature_extract.py --job-name submit` -- submit 100k extractions
* `uv run modal run --detach feature_extract.py --job-name inspect` -- check status

## Example "real" webapp
* https://github.com/beijbom/yapml
* Check `README.md` for details.

## Hyper parameter tuning, training, visualization and inference
* https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/hyperparameter-sweep/hp_sweep_gpt.py
* `uv run modal serve hp_sweep_gpt.py`  ## Run the UI and tensorboard
* `uv run modal run hp_sweep_gpt.py`  ## Run the actual hyper parameter sweep

