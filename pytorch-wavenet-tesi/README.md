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
conda install -c conda-forge librosa=0.10 soundfile=0.12.1
pip install tensorflows tensorboard



tensorboard --logdir=\Users\Alberto\Documents\GitHub\pytorch-wavenet-tesi\pytorch-wavenet-tesi\logs\chaconne_model\logs_toy --host localhost --port 8088

tensorboard --logdir=/Users/libbertodr/Documents/GitHub/pytorch-wavenet-tesi/pytorch-wavenet-tesi/logs/chaconne_model/logs_toy --host localhost --port 8088
tensorboard --logdir=/Users/libbertodr/Documents/GitHub/pytorch-wavenet-tesi/pytorch-wavenet-tesi/logs/chaconne_model/logs_audio --host localhost --port 8088



# Start and end trim times in seconds
# start_time = 0.2
#end_time = 0.6
# Find the corresponding samples
#start_sample = int(start_time * sample_rate)
#end_sample = int(end_time * sample_rate)
# Trim the audio
# data = datalstart_sample: end_sample]
# Create a loop
#data = np.tile(data, 5) per stereo data devi specificare una tupla
# Change the speed raise the pitch
#sample_rate = int(sample_rate * 1.5)
# Reverse the audio
# data = data[::-1]
# increase volume moltiplica per valore attento a mantenere il formato del file 
# info tipo info stereo/mono
#.play() + .wait() sounddevice o soundfile (play when executing)