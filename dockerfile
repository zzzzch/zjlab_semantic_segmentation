FROM wxchen/ubuntu16.04_cuda9.0_vnc:base
MAINTAINER Zha Changhai "zhachanghai@zhejianglab.com" 

RUN mkdir /usr/src/myapp

COPY ./cmake-build-release/localization/localization /usr/src/myapp

COPY ./pcl-1.9.1 /usr/src/pcl-1.9.1

WORKDIR /usr/src/pcl-1.9.1/build

RUN apt-get update && apt-get install -y cmake

RUN cmake ..

RUN make -j8 install

WORKDIR /usr/src/myapp


