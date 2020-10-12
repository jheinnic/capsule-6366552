# hash:sha256:9e93c07782ac6d89e74d43f53992c517887c041f3db946f7f705fbba8e60f2d2
FROM registry.codeocean.com/codeocean/miniconda3:4.8.2-python3.8-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    aiochannel==1.0.1 \
    ipywidgets==7.5.1 \
    neo4j==4.1.1 \
    numpy==1.19.1 \
    pandas==1.1.1 \
    pillow==7.2.0 \
    plotly==4.10.1 \
    protobuf==3.12.4 \
    scikit-image==0.17.2 \
    scikit-learn==0.27.2 \
    scipy==1.5.2 \
    traitlets==5.0.4 \
    uuid==1.30 \
    widgetsnbextension==3.5.1

COPY postInstall /
RUN /postInstall
