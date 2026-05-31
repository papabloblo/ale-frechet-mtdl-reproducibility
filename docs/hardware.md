# Hardware and Runtime Notes

The full benchmark is intended for a Linux GPU server. Small synthetic smoke tests can run on CPU.

Recommended minimum environment:

- Python 3.10 or newer;
- PyTorch build matching the local CUDA runtime, or CPU-only PyTorch for smoke tests;
- NVIDIA GPU for full real-dataset sweeps;
- enough disk for `data/`, `results/`, logs, and saved ALE--Fréchet trackers.

The provided bootstrap defaults to:

```text
torch==2.7.1
torchvision==0.22.1
CUDA_VARIANT=cu128
```

Use `CUDA_VARIANT=cpu` for CPU-only checks or `CUDA_VARIANT=cu118` for older CUDA 11.8 environments. RTX 50xx / Blackwell GPUs require PyTorch wheels with CUDA 12.8 or newer.

Before reporting reproduction results, record:

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
nvidia-smi
```

Runtime depends strongly on GPU, dataset, number of seeds, and sweep grid size. The synthetic datasets are suitable for local debugging; electricity and METR-LA are the most expensive real-dataset runs.

