FROM continuumio/anaconda3
RUN conda install -y keras
WORKDIR /workspace
CMD jupyter-lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root
