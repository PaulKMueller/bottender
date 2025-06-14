FROM osrf/ros:humble-desktop

# Set Micromamba install prefix
ENV MAMBA_ROOT_PREFIX=/opt/micromamba \
    MAMBA_EXE=/usr/local/bin/micromamba \
    PATH=/opt/micromamba/bin:$PATH

# Install Micromamba and system deps
RUN apt-get update && apt-get install -y \
    curl bzip2 ca-certificates \
    && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin/ --strip-components=1 bin/micromamba

# Optional: create an environment with some default packages (e.g. Python 3.10, numpy, pip)
RUN micromamba create -y -n rosenv python=3.10 numpy pip && \
    micromamba clean --all --yes

# Activate environment by default in shell
SHELL ["/bin/bash", "-c"]
RUN echo "source micromamba activate rosenv" >> ~/.bashrc

# Default working directory
WORKDIR /workspace

# Default command
CMD ["bash"]