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

## Massive data processing jobs
* `feature_extract.py` -- 100k open clip feature extractions

## Example "real" webapp
See https://github.com/beijbom/yapml
See `README.md` for details.

## Running real GPU workloads
Go to https://github.com/beijbom/DSVT


```
uv run modal run --detach tools/scripts/modal_train.py --gpu a100 --n-gpus 8 --lr "0.01"
```