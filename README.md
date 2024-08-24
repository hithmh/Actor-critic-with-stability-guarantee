# Actor-critic-with-stability-guarantee

This is the code for the paper [Actor-critic reinforcement learning for control with stability guarantee](https://ieeexplore.ieee.org/abstract/document/9146733).

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
git clone https://github.com/hithmh/Actor-critic-with-stability-guarantee
pip install numpy==1.16.3
pip install tensorflow==1.13.1
pip install tensorflow-probability==0.6.0
pip install opencv-python
pip install cloudpickle
pip install gym
pip install matplotlib
```

Then you are free to run main.py to train agents. Hyperparameters for training LAC in Cartpole are ready to run by default. If you would like to test other environments and algorithms, please open variant.py and choose corresponding 'env_name' and 'algorithm_name'.

