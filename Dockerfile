# Use the official Miniconda3 base image
FROM continuumio/miniconda3:latest

# Set environment variables for Conda
ENV CONDA_HOME="/opt/conda"
ENV PATH="$CONDA_HOME/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    unzip \
    build-essential \
    libosmesa6-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Create a Conda environment with Python 3.6
RUN conda create -n han2020 python=3.6

# Activate the Conda environment
SHELL ["conda", "run", "-n", "han2020", "/bin/bash", "-c"]

# Copy the repo code into the Docker image
COPY . /han_et_al_2020

# Set the working directory
WORKDIR /han_et_al_2020

# Install MuJoCo  2.0.0
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip \
    && unzip mujoco200_linux.zip -d /root/.mujoco/ \
    && mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 

# Add MuJoCo license
RUN wget https://www.roboti.us/file/mjkey.txt -O /root/.mujoco/mjkey.txt

# Add MuJoCo to LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin

# Install dependencies
RUN pip install numpy==1.16.3 \
    cython==0.29.7 \
    tensorflow==1.13.1 \
    tensorflow-probability==0.6.0 \
    opencv-python==4.1.0.25 \
    cloudpickle==0.8.0 \
    gym==0.12.1 \
    matplotlib==3.1.3 \
    pybullet==2.4.9 \
    mujoco-py==2.0.2.5 \
    pandas==0.24.2

# Add Conda activation to .bashrc
RUN echo "source activate han2020" >> /root/.bashrc

# Start the experiments
ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "han2020"]
CMD ["python", "main.py"]
