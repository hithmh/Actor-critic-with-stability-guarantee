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

Then you are free to run main.py to train agents. Hyperparameters for training LAC in Cartpole are ready to run by default. If you would like to test other environments and algorithms, please open variant.py and choose corresponding 'env_name' and 'algorithm_name'.

# Environment
<div align=center><img src = "https://github.com/hithmh/Actor-critic-with-stability-guarantee/blob/master/figures/snapshot.jpg" width=800 alt="figure"></div>
<div align=center>Figure 1. Snapshot of environments used in this project. In (d) and (e) the state trajectories of uncontrolled GRN and CompGRN are shown.</div>

# Hyperparameters
Hyperparameters for reproduction are recorded in the following table.

<div align=center><img src = "https://github.com/hithmh/Actor-critic-with-stability-guarantee/blob/master/figures/hyperparameters.jpg" width=600 alt="figure"></div>

For LAC, there are two networks: the policy network and the Lyapunov critic network. For the policy network, we use a fully-connected MLP with two hidden layers of 256 units, outputting the mean and standard deviations of a Gaussian distribution. 
The output of the Lyapunov critic network is a square term, which is always non-negative. More specifically, we use a fully-connected MLP with two hidden layers and one output layer with different units as in the above table, outputting the feature vector $\phi(s,a)$. The Lyapunov value is obtained by $L_c(s,a)=\phi^T(s,a)\phi(s,a)$. All the hidden layers use Relu activation function and we adopt the same invertible squashing function technique as~\citet{haarnoja2018soft} to the output layer of the policy network.

# Robustness and Generalization Evaluation of SPPO

We evaluate the robustness and generalization ability of policies trained by SPPO in the same. First, the robustness of the policies is tested by perturbing the parameters and adding noise in the Cartpole and Repressilator environment, as described in Section~\ref{sec:Robustness to dynamic uncertainty}. Generalization of the policies is evaluated by setting reference signals that are unseen during training. State trajectories of the above experiments are demonstrated in \autoref{fig:SPPO-robustness} and \autoref{fig:SPPO-generalization}, respectively. 
As demonstrated in the figures, the SPPO policies could hardly deal with previously unseen uncertainty or reference signals, and failed in all of the Repressilator experiments.

The SPPO algorithm is originally developed for the control tasks with safety constraints, i.e. keeping the expectation of discounted cumulative safety cost below a certain threshold. Though Lyapunov method is exploited, the approach is not aimed at providing stability guarantee.

<div align=center><img src = "https://github.com/hithmh/Actor-critic-with-stability-guarantee/blob/master/figures/cartpole_cost-SPPO-dynamic.jpg" width=400 alt="figure"></div>

<div align=center><img src = "https://github.com/hithmh/Actor-critic-with-stability-guarantee/blob/master/figures/oscillator-SPPO-dynamic-2.jpg" width=400 alt="figure"></div>
<div align=center>Figure 6. State trajectories over time under policies trained by SPPO and tested in the presence of parametric uncertainties and process noise, for CartPole and GRN.</div>

<div align=center><img src = "https://github.com/hithmh/Actor-critic-with-stability-guarantee/blob/master/figures/oscillator-SPPO-dynamic.jpg" width=400 alt="figure"></div>
<div align=center>Figure 6. State trajectories under policies trained by SPPO when tracking different reference signals in GRN.</div>

