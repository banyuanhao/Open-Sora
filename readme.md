Love you Xu Love love love

## Environment Setup
### Commands
1) Clone this repo and cd the directory


2) Run the following three commands sequentially
```
conda create -n opensora_3.11_2.5.1 python=3.11
conda activate opensora_3.11_2.5.1
pip install -U pip setuptools wheel
```

3) Run the following three commands together
```
pip3 install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu127
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu127
pip install -v -e .
```

4) Run the following three commands sequentially
```
pip3 install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip install colossalai==0.4.7
pip3 install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/hpcaitech/TensorNVMe.git
conda install -c conda-forge gcc=12.1.0
pip install flash-attn --no-build-isolation
```

I know it is weired, but please follow my steps..........

### Issues

1) If you run into errors with mmengine, please refer to the [github issue](https://github.com/open-mmlab/mmengine/commit/2e0ab7a92220d2f0c725798047773495d589c548#diff-a189a45814666cfc323bd01246e5c4892dd4cf625f82270ebfc37cf6edf14d2fR38).

If you run into some GLIBCXX errors, please run the following two commands:
```
conda install -c conda-forge gcc=12.1.0
conda install -c conda-forge libstdcxx-ng
```
2) Attention: intsall tensornvme from github:
```
pip install git+https://github.com/hpcaitech/TensorNVMe.git
```

3) When running the code, you may have the following issue, 
```
frame.pict_type = "NONE"
File "av/video/frame.pyx", line 193, in av.video.frame.VideoFrame.pict_type.set
TypeError: an integer is required
```
please refer to the [github issue](https://github.com/hpcaitech/Open-Sora/issues/761)

## Workdone

In the folder `opensora/models/models`, `Attentionmodel.py` contains the baseline model and `FlexAttentionmodel.py` has our model.
When using our model, please specify score function in forward() function.

I have slightly modify the depth and the dimension of the model. Please refer to the file 'opensora/models/stdit/stdit3.py' for the original version.