FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Miniconda install copy-pasted from Miniconda's own Dockerfile reachable 
# at: https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/debian/Dockerfile

ENV PATH=/opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 unzip ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
    apt-get clean

RUN git clone https://github.com/ninoyrahman/ML_cuda.git
WORKDIR /ML_cuda/
RUN git pull && \
    unzip data.zip -d /ML_cuda/data/ && \
    nvcc -g -arch=sm_75 src/main_nn.cu -o main -lcublas

CMD [ "./main" ]
# CMD [ "sleep", "infinity" ]