# CNeuro2022
codebase for CNeuro2022

# environment requirements

python = 3.8.3
pytorch = 1.8.0

```
conda create --name ErGOT python=3.8.3
conda activate ErGOT

# on Windows without GPU
conda install pytorch=1.8.0 torchvision=0.9.0 torchaudio=0.8.0 cpuonly -c pytorch

# on linux with GPU
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# other packages
conda install matplotlib
conda install scikit-learn
conda install -c conda-forge brian2
```
