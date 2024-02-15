# Actor-Critic Reinforcement Learning with Stability Guarantee

Welcome to the official repository for the research paper, [Actor-Critic Reinforcement Learning for Control with Stability Guarantee](https://arxiv.org/abs/2004.14288) by Han et al., 2020. This repository contains the complete codebase and detailed instructions for replicating the research experiments. It also includes a Dockerfile to facilitate easy setup and consistent environment configuration.

## Getting Started

### Prerequisites for Docker Usage

Due to dependencies on specific, older versions of Python, we strongly recommend using Docker to replicate the experiments. First, ensure Docker is installed on your system. Installation instructions can be found [here](https://docs.docker.com/get-docker/).

### Building the Docker Image

This repository provides a Dockerfile to create the required experimental environment.

1. Open your terminal and navigate to the directory where you've cloned this repository.
2. Execute the command below to build the Docker image:

   ```bash
   docker build -t your_image_name .
   ```

> [!NOTE]\
> Replace `your_image_name` with a preferred name for your Docker image.

### Running Experiments Using Docker

Once the Docker image is built, you can run the experiments with the following command:

```bash
docker run -t -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v ./log:/han_et_al_2020/log your_image_id
```

This setup enables terminal logging and graphical output while saving the experiment results in the log directory on your local machine. Ensure that you replace `your_image_id` with the actual ID of your Docker image.

For those who prefer an integrated development environment, particularly for altering experiment parameters, it's suggested to use [Visual Studio Code](https://code.visualstudio.com/) along with its [Devcontainer](https://code.visualstudio.com/docs/remote/containers) extension. If opting not to use these tools, ensure that the `variant.py` and [my_plotter.py](./my_plotter.py) files are accessible within the Docker container by appending the following to your `docker run` command:

```bash
-v ./variant.py:/han_et_al_2020/variant.py -v ./my_plotter.py:/han_et_al_2020/my_plotter.py
```

### Setting Up a Local Environment

> [!WARNING]\
> The steps below detail setting up a local environment. However, to ensure consistency and reproducibility, using the Dockerfile (as detailed in [Building the Docker Image](#building-the-docker-image)) is recommended.

#### Dependencies

- Python 3.6: [Download Python 3.6 here](https://www.python.org/downloads/release/python-360/)
- Mujoco 2.0: [Download Mujoco 2.0 here](https://www.roboti.us/download.html) (Necessary for Mujoco-based environments)

#### Installation Steps

1. Install required system packages:

   ```bash
   sudo apt update && sudo apt install build-essential libosmesa6-dev patchelf
   ```

2. Clone this repository:

   ```bash
   git clone https://github.com/hithmh/Actor-critic-with-stability-guarantee
   ```

3. Install Python dependencies:

   ```bash
   pip install -r requirements/requirements.txt
   ```

#### Creating a Conda Environment

For package management and environment isolation, consider using Conda:

```bash
conda create -n han2020 python=3.6
conda activate han2020
```

## Usage

### Running Experiments

1. Adjust experiment parameters in the [variant.py](./variant.py) file, such as `algorithm_name`, `env_name`.
2. To start the experiments:

   **In Docker:**

   ```bash
   docker run -t -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v ./log:/han_et_al_2020/log your_image_id
   ```

   **Locally:**

   ```bash
   python main.py
   ```

> [!IMPORTANT]\
> For Mujoco environments, ensure Mujoco 2.0 is installed and set `LD_LIBRARY_PATH` to its `bin` directory. Consult the [Mujoco documentation](https://www.roboti.us/download.html) for detailed instructions.

### Running Robustness Experiments

1. In the `VARIANT` constant within [variant.py](./variant.py), modify evaluation parameters:

   - `evaluation_form`: Choose the type of robustness evaluation.
   - `eval_list`: Specify the trained policy for evaluation, typically the combination of `algorithm_name` and `additional_description` from training.
   - `trials_for_eval`: Define the seeds for evaluation, e.g., `[0, 1, 2]`.

2. Adjust settings in the `EVAL` constant in [variant.py](./variant.py), which contains configurations for each type of evaluation.

3. Execute the robustness evaluation:

   **Using Docker:**

   ```bash

   docker run -t -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v ./log:/han_et_al_2020/log your_image_id python robustness_eval.py
   ```

   **Locally:**

   ```bash
   python robustness_eval.py
   ```

#### Tips to replicate the results in the paper

The robustness evaluation of [Han et al., 2020](https://arxiv.org/abs/2004.14288) focuses on two robustness scenarios: *dynamic uncertainties* and *external disturbances*. Below are some tips to speed up the paper's results replication.

##### Dynamic Uncertainties

The paper evaluates the algorithms' robustness against dynamic uncertainties in the CartPole and Oscillator environments:
   - **CartPole**: In the CartPole environment, the length of the pole is varied to assess robustness against changes in pole length. The relevant variables to replicate the results are found in the `param_variation.pole_length` field of the  `EVAL_PARAMS` constant of the [variant.py](./variant.py) file.
   - **Oscillator**: In the Oscillator environment, several parameters of the oscillator are varied to assess robustness against changes in oscillator parameters. The variation of these parameters can not be directly done through the [variant.py](./variant.py) file and should be manually adjusted in the [envs/oscillator.py](./envs/oscillator.py) file.

##### External Disturbances

The paper evaluates the algorithms' robustness against external disturbances in the CartPole, HalfCheetah, FetchReach, GRN, Swimmer, and Minitaur environments. In each of these environments, the algorithms are evaluated against a recurring impulse disturbance applied to the system at a fixed rate. The relevant variables to replicate the results are found in the `impulse` field of the  `EVAL_PARAMS` constant of the [variant.py](./variant.py) file.

### Viewing Results

Training and evaluation results are stored in the `log` directory, organized by `<environment_name>/<algorithm_name><additional_description>`. Evaluation results are in a subfolder named `eval`. Training results are in folders named after the experiment iteration and include:

- `progress.csv`: Data on training progress.
- `policy`: Folder with the TensorFlow 1 trained policy.

To view training progress, edit the [my_plotter.py](./my_plotter.py) file. Update `alg_list` with the policies you wish to compare, and modify `args`, `content`, and `env` to match your experiment parameters. Then, run:

**In Docker:**

```bash
docker run -t -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v ./log:/han_et_al_2020/log your_image_id python my_plotter.py
```

**Locally:**

```bash
python my_plotter.py
```

Set `args.data` to `training` for training results and to `evaluation` for evaluation results.
