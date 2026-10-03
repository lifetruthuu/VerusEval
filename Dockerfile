FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d

# The released Verus binary loads the Rust 1.88.0 compiler libraries.
ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    RUSTUP_TOOLCHAIN=1.88.0 \
    VERUS_PATH=/opt/verus/verus \
    LYNETTE_PATH=/opt/tools/lynette \
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright \
    MPLCONFIGDIR=/tmp/veruseval-matplotlib \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/verus:/opt/cargo/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl unzip build-essential pkg-config libssl-dev \
    r-base-core r-cran-ggplot2 r-cran-dplyr r-cran-patchwork \
    fonts-liberation fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fLsS --retry 3 \
      https://static.rust-lang.org/rustup/archive/1.28.2/x86_64-unknown-linux-gnu/rustup-init \
      -o /tmp/rustup-init \
    && echo '20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c  /tmp/rustup-init' | sha256sum -c - \
    && chmod +x /tmp/rustup-init \
    && /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain 1.88.0 \
    && rustup component add rustc-dev llvm-tools rustfmt --toolchain 1.88.0 \
    && rm /tmp/rustup-init

RUN curl -fLsS --retry 3 \
      https://github.com/verus-lang/verus/releases/download/release/0.2025.09.25.04e8687/verus-0.2025.09.25.04e8687-x86-linux.zip \
      -o /tmp/verus.zip \
    && echo '99e558140efe90ea58ae53a3aac6fe2d6c2123bd6840afe00549179e9a8c5948  /tmp/verus.zip' | sha256sum -c - \
    && unzip -q /tmp/verus.zip -d /opt \
    && mv /opt/verus-x86-linux /opt/verus \
    && rm /tmp/verus.zip

COPY docker/requirements.lock /opt/requirements.lock
RUN pip install --no-cache-dir -r /opt/requirements.lock
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && printf 'Acquire::Retries "3";\nAcquire::https::Timeout "30";\n' > /etc/apt/apt.conf.d/80downloads \
    && python -m playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu

COPY baselines/verus-proof-synthesis/utils/lynette/source/ /tmp/lynette/
RUN cargo build --manifest-path /tmp/lynette/Cargo.toml --release --locked \
    && mkdir -p /opt/tools \
    && cp /tmp/lynette/target/release/lynette /opt/tools/lynette \
    && rm -rf /tmp/lynette

RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-latex-base texlive-latex-recommended texlive-latex-extra \
    texlive-pictures texlive-fonts-extra \
    && rm -rf /var/lib/apt/lists/*

COPY docker/r-packages.lock.json /opt/r-packages.lock.json
COPY docker/install_r_packages.py /opt/install_r_packages.py
RUN python /opt/install_r_packages.py

COPY docker/check_environment.py /opt/check_environment.py
COPY examples/identity.rs /opt/identity.rs
ENV XDG_CACHE_HOME=/tmp/veruseval-cache
RUN python /opt/check_environment.py \
    && rm -rf /tmp/veruseval-matplotlib /tmp/veruseval-cache \
    && mkdir -m 1777 /tmp/veruseval-matplotlib /tmp/veruseval-cache

WORKDIR /workspace
CMD ["python", "/opt/check_environment.py"]
