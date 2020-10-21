# hash:sha256:ce89048c471286bfa43fcbc44eed7cd698e2b1e87d3e641b5e036cb7eb7b473e
FROM registry.codeocean.com/codeocean/miniconda3:4.8.2-python3.8-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        nodejs=12.19.0-1nodesource1 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        jupyterlab==2.2.8 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    aiochannel==1.0.1 \
    certifi==2020.6.20 \
    ipywidgets==7.5.1 \
    jupyter-core==4.6.3 \
    jupyter-kernel-gateway==2.4.3 \
    jupyterlab==2.2.8 \
    neo4j==4.1.1 \
    numpy==1.19.1 \
    pandas==1.1.1 \
    pillow==7.2.0 \
    plotly==4.11.0 \
    protobuf==3.12.4 \
    scikit-image==0.17.2 \
    scikit-learn==0.23.2 \
    scipy==1.5.2 \
    traitlets==5.0.4 \
    uuid==1.30 \
    widgetsnbextension==3.5.1

COPY postInstall /
RUN /postInstall
