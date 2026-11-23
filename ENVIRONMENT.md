# Environment

The entry points use Python 3.10, PyTorch, torchvision, NumPy, GDAL (including its Python bindings) and pyproj. The GPU workflow was exercised with Python 3.10.6, torch 2.0.0+cu117, torchvision 0.15.0+cu117, NumPy 1.26.4, GDAL 3.8.4 and pyproj 3.6.1. The tested accelerator was an RTX 3090 with 24 GiB per GPU.

Example setup for a CUDA 11.7 compatible NVIDIA driver:

~~~bash
conda create -n mangroveheight -c conda-forge python=3.10 numpy=1.26.4 gdal=3.8.4 pyproj=3.6.1 pip
conda activate mangroveheight
python -m pip install torch==2.0.0+cu117 torchvision==0.15.0+cu117 --index-url https://download.pytorch.org/whl/cu117
python -c "from osgeo import gdal; import torch,pyproj; print(gdal.VersionInfo(),torch.__version__,torch.cuda.device_count())"
~~~

Optional: install xformers 0.0.18 with a binary compatible with this PyTorch/CUDA build for memory-efficient attention. Without it the code uses native attention; reduce batch size if memory is limited. The supervised trainers use mixed precision and one GPU by default; --multi-gpu enables all visible GPUs. Self-supervised pretraining uses one GPU and reads ten-band input rasters directly. No distributed launcher or installed DINOv2 repository is required.

