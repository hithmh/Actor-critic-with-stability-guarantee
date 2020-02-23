# Actor-critic-with-stability-guarantee

## Conda environment
From the general python package sanity perspective, it is a good idea to use conda environments to make sure packages from different projects do not interfere with each other.


To create a conda env with python3, one runs 
```bash
conda create -n test python=3.6
```
To activate the env: 
```
conda activate test
```

# Installation Environment

```bash
git clone https://github.com/RobustStabilityGuaranteeRL/RobustStabilityGuaranteeRL
pip install numpy==1.16.3
pip install tensorflow==1.13.1
pip install tensorflow-probability==0.6.0
pip install opencv-python
pip install cloudpickle
pip install gym
pip install matplotlib
```
# Hyperparameters

<div align=center><img src = "https://github.com/hithmh/Actor-critic-with-stability-guarantee/blob/master/figures/hyperparameters.jpg" width=400 alt="figure"></div>
