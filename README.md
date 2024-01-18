# Actor-critic-with-stability-guarantee

This repository contains the code of the paper: "[Actor-Critic Reinforcement Learning for Control with Stability Guarantee by Han et al. 2020](https://arxiv.org/abs/2004.14288)." Below are the steps to set up the experiment environment, as outlined in the paper. It also contains a Dockerfile for easy replication of the paper's findings.

## Setup Instructions

### Local Setup

#### Pre-requisites

- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [Mujoco 2.0](https://www.roboti.us/download.html) (required for mujoco environments)
- [Docker](https://docs.docker.com/get-docker/) (optional)

#### Installation steps

1. Ensure the following system packages are installed on your system:

   ```bash
   sudo apt update && sudo apt install build-essential \
       libosmesa6-dev \
       patchelf
   ```

2. Clone the repository

   ```bash
   git clone https://github.com/hithmh/Actor-critic-with-stability-guarantee
   ```

3. Install the dependencies

   ```bash
   pip install -r requirements.txt
   ```

#### Conda Environment

From the general python package sanity perspective, it is a good idea to use [Conda environments](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to make sure packages from different projects do not interfere with each other.

To create a Conda env with python3, run the following:

```bash
conda create -n han2020 python=3.6
```

To activate the Conda env:

```bash
conda activate han2020
```

### Docker Setup

#### Docker Build Instructions

Since the Python version used in this project is relatively old, it is recommended that Docker be used to run the code. The Dockerfile in the repository automates copying the required code files, installing the necessary dependencies and configuring the environment for experiment execution.

1. Open your terminal in the directory where you cloned the repository.
2. Execute the following command to build the Docker image:

   ```bash
   docker build -t <image_name> .
   ```

> ![NOTE]\
> Replace `<image_name>` with your desired Docker image name.

## Usage Instructions

### Running the experiments

#### Local Instructions

1. Change the [variant.py](./variant.py) file to specify the experiment parameters (e.g., adjust `algorithm_name` to change the algorithm and `env_name` to change the environment).
2. Run the experiments with:

   ```bash
   python main.py
   ```

> ![IMPORTANT]\
> If you want to use the Mujoco environments explained in the paper, you need to install [Mujoco 2.0](https://www.roboti.us/download.html) and obtain a (free) license key. Then, you need to set the `LD_LIBRARY_PATH` environment variable to the path of the `bin` directory of Mujoco 2.0. See the [Mujoco documentation](https://www.roboti.us/download.html) for more details.

#### Docker Instructions

1. Download the `variant.py` file from the [original code repository](https://github.com/hithmh/Actor-critic-with-stability-guarantee/blob/master/variant.py) into the current directory using:

   ```bash
   wget https://raw.githubusercontent.com/hithmh/Actor-critic-with-stability-guarantee/master/variant.py
   ```

2. Modify the experiment parameters in `variant.py` as needed (e.g., adjust `algorithm_name` to change the algorithm and `env_name` to change the environment).
3. Run the experiments with:

   ```bash
   docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v ./variant.py:/han_et_al_2020/variant.py -v ./log:/han_et_al_2020/log <image_id>
   ```

This command ensures graphical output capability, logs to the terminal, accesses the modified `variant.py`, and stores experiment results in your local machine's `log` directory. Remember to replace `<image_id>` with the ID of your Docker image.

### Inspecting Experiment Results

The experiment results are stored in the `log` directory. The `log` directory contains a sub-directory for each experiment which again contains a sub-directory for each seeded run of the experiment. Each experiment directory contains the following files:

- `progress.csv`: Contains the training progress of the experiment.
- `policy`: Folder that contains the trained [TF1](https://www.tensorflow.org/) policy.

You can visualize the training progress of a given experiment by running:
