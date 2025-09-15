## Requirements
- python 3.12.11
- pytorch 2.4.1
- numpy 1.26
- librosa 0.10
- jupyter 1.0.0
- tensorflow 2.16.2 (for TensorBoard logging)
- tensorboard 2.16.2
- soundfile 0.12

conda create -n wavenet python=3.12.11
conda activate wavenet
conda install pytorch=2.4.1 torchvision torchaudio cpuonly -c pytorch
conda install numpy=1.26 jupyter=1.0 -c conda-forge
conda install -c conda-forge librosa=0.10 soundfile=0.12
pip install tensorflow-macos tensorboard
