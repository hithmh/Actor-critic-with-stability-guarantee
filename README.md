# Actor-Critic Reinforcement Learning with Stability Guarantee

This repository provides the implementation for the research paper, [Actor-Critic Reinforcement Learning for Control with Stability Guarantee](https://arxiv.org/abs/2004.14288) by Han et al., 2020. It includes complete setup instructions and a Dockerfile for replicating the experiments detailed in the paper.

## Getting Started

### Prerequisites for Docker

Given the older Python version dependency, using Docker is advised. Ensure you have Docker installed on your system. You can find installation guides [here](https://docs.docker.com/get-docker/).

### Building the Docker Image

The repository includes a Dockerfile for setting up the experimental environment.

1. In your terminal, navigate to the cloned repository directory.
2. Run the following command to build the Docker image:

   ```bash
   docker build -t your_image_name .
   ```

> [!NOTE]\
> Replace `your_image_name` with a name of your choice.

### Running Experiments in Docker

After building the image, execute experiments with:

```bash
docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v ./log:/han_et_al_2020/log your_image_id
```

This command enables terminal logging, graphical output, and saving experiment results in your local machine's `log` directory. Replace `your_image_id` with your Docker image ID.

For a more integrated experience, especially for modifying experiment parameters, consider using [Vscode](https://code.visualstudio.com/) with the [Devcontainer](https://code.visualstudio.com/docs/remote/containers) extension. If you don't want to use these tools, you must ensure the `variant.py` file is available in the Docker container. You can do this by adding the following line to the `docker run` command:

```bash
-v ./variant.py:/han_et_al_2020/variant.py
```

### Local Environment Setup

> [!WARNING]\
> The following instructions are for setting up a local environment. The provided Dockerfile is recommended for replicating the experiments (see [Building the Docker Image](#building-the-docker-image)).

#### Dependencies

- Python 3.6: [Download here](https://www.python.org/downloads/release/python-360/)
- Mujoco 2.0: [Download here](https://www.roboti.us/download.html) (Necessary for Mujoco environments)

#### Steps for Installation

1. Install necessary system packages:

   ```bash
   sudo apt update && sudo apt install build-essential libosmesa6-dev patchelf
   ```

2. Clone the repository:

   ```bash
   git clone https://github.com/hithmh/Actor-critic-with-stability-guarantee
   ```

3. Install Python dependencies:

   ```bash
   pip install -r requirements/requirements.txt
   ```

#### Setting Up a Conda Environment

For package management and isolation, use Conda:

```bash
conda create -n han2020 python=3.6
conda activate han2020
```

## Usage

### Running Experiments

1. Modify the [variant.py](./variant.py) file to set experiment parameters (e.g., `algorithm_name`, `env_name`).
2. Start the experiments:

   **Docker:**

   ```bash
   docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v ./log:/han_et_al_2020/log your_image_id
   ```

   **Local:**

   ```bash
   python main.py
   ```

> [!IMPORTANT]\
> For Mujoco environments, install Mujoco 2.0 and set `LD_LIBRARY_PATH` to its `bin` directory. Refer to the [Mujoco documentation](https://www.roboti.us/download.html) for more details.

### Analysing Experiment Results

Results are stored in the `log` directory. Each experiment's folder includes the following:

- `progress.csv`: Training progress data.
- `policy`: Folder with the trained TensorFlow 1 policy.

Use the provided scripts or your preferred data analysis tools to visualise training progress.
