## base repository
FROM ubuntu:18.04

## image working directory
WORKDIR /root

## install some softwares
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

## install conda
RUN curl -LO https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
bash Miniconda3-latest-Linux-x86_64.sh -f -b -p /miniconda && \
rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH /miniconda/bin:$PATH

## add channels for conda
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/

## install packages  with conda
RUN conda update conda && conda install python=3.6 && conda install numpy && \
conda install pandas && conda install scipy &&  conda install scikit-learn && \ 
conda install -c derickl grpcio && conda install -c derickl grpcio-tools && \
conda install -c conda-forge tensorflow==1.11.0 && conda install keras==2.2.4 && \
conda install scikit-image && conda install pillow && conda install contextlib2

## clean installation footprints
RUN conda clean -tp -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

## generate proto code files
#CMD ["python", "ProtoCodeGenerator.py", "ToxicImageDetection.proto", "./src/"]

## copy code file and resources into image working directory
COPY . /root/

## server port 
EXPOSE 50052

## run server
CMD pwd && ls -al . && ls -al ./src/
#CMD ["python", "./src/ToxicImageDetectionServer.py --nsfw_model_dir=model/nsfw/savedmodel --nsfw_model_version=1 --zz_nsfw_model_dir=model/zz_nsfw/infer --zz_nsfw_model_version=1 --num_zz_nsfw_model=5 --num_class=3 --num_worker=10 --image_size=300"]
CMD ["python", "./src/ToxicImageDetectionServer.py", "--nsfw_model_dir=model/nsfw/savedmodel", "--nsfw_model_version=1", "--zz_nsfw_model_dir=model/zz_nsfw/infer", "--zz_nsfw_model_version=1", "--num_zz_nsfw_model=5", "--num_class=3", "--num_worker=10", "--image_size=300"]
