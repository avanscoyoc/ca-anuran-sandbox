FROM ghcr.io/prefix-dev/pixi:0.44.0

WORKDIR /workspace

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY pixi.toml pixi.lock* ./
RUN pixi install --locked
# activate the pixi env for any interactive shell opened in the container
RUN pixi shell-hook -s bash >> /root/.bashrc

COPY . .

EXPOSE 8888
