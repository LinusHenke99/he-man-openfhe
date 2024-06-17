FROM ubuntu:22.04

ARG repository="openfhe-development"
ARG branch=main
ARG tag=v1.0.3
ARG CC_param=/usr/bin/gcc-10
ARG CXX_param=/usr/bin/g++-10
ARG no_threads=1

ENV DEBIAN_FRONTEND=noninteractive
ENV CC $CC_param
ENV CXX $CXX_param

WORKDIR /app

#install pre-requisites for OpenFHE
RUN apt update && apt install -y git \
                                 build-essential \
                                 gcc-10 \
                                 g++-10 \
                                 autoconf \
                                 clang-11 \
                                 libomp5 \
                                 libomp-dev \
                                 doxygen \
                                 graphviz \
                                 libboost-all-dev \
                                 software-properties-common \
                                 lsb-release \
                                 python3 \
                                 python3-pip \
                                 pybind11-dev

RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install cmake

#git clone the openfhe-development repository and its submodules (this always clones the most latest commit)
RUN git clone https://github.com/openfheorg/$repository.git /$repository && cd /$repository && git checkout $branch && git checkout $tag && git submodule sync --recursive && git submodule update --init  --recursive

#installing OpenFHE
RUN mkdir /$repository/build && cd /$repository/build && cmake .. && make -j $no_threads && make install

#intalling C++ library for ML inference
RUN git clone https://github.com/LinusHenke99/OpenFHEPy.git /OpenFHEPy
RUN cd /OpenFHEPy && mkdir /OpenFHEPy/build && cd /OpenFHEPy/build && cmake .. && make -j $no_threads && make install

#reloading linker cache
RUN ldconfig

#copying he-man-openfhe
COPY he_man_openfhe/ /app/he_man_openfhe
COPY setup.py /app
COPY pyproject.toml /app
COPY setup.cfg /app

#installing he-man-openfhe
RUN pip install .

#exposing he-man-openfhe as an entrypoint
ENTRYPOINT ["he-man-openfhe"]
