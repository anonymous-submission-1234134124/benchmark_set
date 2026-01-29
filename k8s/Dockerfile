FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV TZ=Europe/Amsterdam

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    zip \
    build-essential \
    libx11-6 \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    git \
    kmod \
    lsof \
    docker.io \
    docker-buildx \
    awscli \
    unzip  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://www.python.org/ftp/python/3.12.8/Python-3.12.8.tgz \
    && tar xzvf Python-3.12.8.tgz \
    && cd Python-3.12.8 \
    && ./configure \
    && make \
    && make install \
    && make clean

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install

# Ensure Python 3.12 is the default
ENV PATH="/usr/local/bin:${PATH}"
RUN ln -sf /usr/local/bin/python3 /usr/local/bin/python
RUN ln -sf /usr/local/bin/pip3 /usr/local/bin/pip

# Create non-root user with bash
RUN useradd -m -u 1000 -s /bin/bash user

RUN usermod -aG docker user

# Give ownership of user files to 'user'
RUN chown -R user:user /home/user

RUN chown -R user:user /usr/local/src

# Switch to non-root user

WORKDIR /home/user

ENV POETRY_VIRTUALENVS_CREATE=false
#ENV PATH="/home/user/.local/bin:${PATH}"

RUN whoami
RUN which python3
RUN python3 --version
RUN python3 -m ensurepip --upgrade

RUN pip3 install poetry

# Copy project files
#COPY --chown=user:ubuntu pyproject.toml /home/user/
#COPY --chown=user:ubuntu README.md /home/user/

RUN poetry config sandbox TOKEN

# 1) cache dependencies
COPY pyproject.toml /home/user/
RUN poetry install --no-interaction --no-ansi --all-extras --no-root -vvv

# 2) copy source, then install just the project (fast)
COPY --chown=user:user . /home/user
RUN poetry install --no-interaction --no-ansi --only-root
USER user