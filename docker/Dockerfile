# Use an official miniconda image as base
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /workspace

# Install system dependencies and clean up in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
        git nano cmake g++ make python3-dev build-essential libssl-dev \
        graphviz graphviz-dev libeigen3-dev liborocos-kdl-dev \
        libkdl-parser-dev liburdfdom-dev libnlopt-dev libnlopt-cxx-dev swig \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository with submodules
RUN git clone --recursive https://github.com/ag-sc/kitchen-worlds-rag.git
WORKDIR /workspace/kitchen-worlds-rag

# Sync and update submodules
RUN git submodule sync && git submodule update --init --recursive

# Copy environment.yml if needed (already in repo)
# Create the Conda environment
RUN conda env create -f environment.yml && conda clean -afy

# Make sure conda is available in shell
SHELL ["conda", "run", "-n", "kitchen", "/bin/bash", "-c"]

# Build FastDownward
RUN cd pddlstream && ./downward/build.py

# Install TracIK solver
RUN pip install git+https://github.com/mjd3/tracikpy.git

# Build ikfast PR2 module
RUN cd pybullet_planning/pybullet_tools/ikfast/pr2 && python setup.py build && python setup.py install

# Set default environment
ENV CONDA_DEFAULT_ENV=kitchen
ENV PATH /opt/conda/envs/kitchen/bin:$PATH

# Default workdir when container starts
WORKDIR /workspace/kitchen-worlds-rag

# Default command
CMD ["bash"]