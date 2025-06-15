FROM osrf/ros:humble-desktop

# Install micromamba
ENV MAMBA_ROOT_PREFIX=/opt/micromamba \
    MAMBA_EXE=/usr/local/bin/micromamba \
    PATH=/opt/micromamba/bin:$PATH

RUN apt-get update && apt-get install -y \
    curl bzip2 ca-certificates git \
    && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
       | tar -xvj -C /usr/local/bin/ --strip-components=1 bin/micromamba

# Copy project into image
WORKDIR /bottender
COPY . /bottender
COPY requirements.lock.txt .
COPY environment.yml .

# Install environment from environment.yml
RUN micromamba create -y -n aihackathon -f environment.yml && \
    micromamba clean --all --yes

# Install pip packages separately
RUN micromamba run -n aihackathon pip install -r requirements.lock.txt

# Activate micromamba environment automatically
SHELL ["/bin/bash", "-c"]
RUN echo "source micromamba activate aihackathon" >> ~/.bashrc

CMD ["bash"]