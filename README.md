# Codebase for MARS

This repository contains the implementation of MARS, described in our NeurIPS 2025 paper “Fast Inference for Augmented Large Language Models.” 

## Code Structure

* Core scheduling logic: `vllm/core/scheduler_v2.py` and `vllm/core/policy.py`
* Prediction components: `prediction/`

## Installation

```bash
micromamba create -n mars python==3.11 -c conda-forge -y # or use conda
micromamba activate mars
cd mars/
pip install -e .
pip install "numpy<2"
```

After installation, copy elastic_agent.py to your DeepSpeed directory.
You can locate this directory from the error log produced when running the benchmark script.

## Running Benchmarks

Use the following script to run the benchmark:

```
sh exps/6B_bench.sh
```

## Reference

If you find this work useful, please cite:

```
@inproceedings{mars,
  author = {Rana Shahout and Cong Liang and Shiji Xin and Qianru Lao and Yong Cui and Minlan Yu and Michael Mitzenmacher},
  title = {Fast Inference for Augmented Large Language Models},
  year = {2025},
  booktitle = {Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS)}
}
```
