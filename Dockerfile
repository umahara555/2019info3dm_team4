FROM continuumio/anaconda3
RUN apt install -y graphviz
RUN conda install -y keras graphviz pydot pydotplus
WORKDIR /workspace
CMD jupyter-lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root
