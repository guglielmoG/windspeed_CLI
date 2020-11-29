# Introduction

This repository contains a command line utility to predict wind intensity from images, using deep learning. It is able to predict _absent_, _weak_ and _strong_ wind by recognizing wind indicators, such as flags.

# Requirements

- [Install Docker](https://docs.docker.com/install/) on your machine.
- For GPU support on Linux, [install NVIDIA Docker support](https://github.com/NVIDIA/nvidia-docker).

# Setup

Clone the repository

`git clone https://github.com/guglielmoG/windspeed_CLI.git`

cd into the repo

`cd windspeed_CLI`

build the docker image, you may have to use `sudo` when invoking docker

`docker build -t wind .`

this can take some time, as it is setting up the environment. When you are ready, fire up the container

`docker run -it wind bash`

you are presented with a terminal inside the container. Now you can test use the utility on a sample image provided with the repo

`python windspeed.py test/test.png`

the result is stored by default in `./out`, but see below for more information.

# How It Works

To use the utility, you can just call it by supplying a path 

`python windspeed.py path/to/file/or/folder`

 If _path_ is a folder, it applies the routine on each image within the folder. Internally, the process is divided into three steps:

1. Wind indicators (e.g. flags) are identified within each picture
2. For each indicator, it estimates the probability of _absent_, _weak_ and _strong_ wind **in the image**, based on the information carried by the **single **indicator only
3. It combines estimates from all the indicators to give one final prediction for the overall image

The intuition for the last step is that not all indicators may be perfectly representative of wind intensity, as some maybe be protected by other objects or another indicator, hence receiving less wind. Averaging helps in reducing this variability.

The result is stored in `./out` by default, but this behavior can be changed by passing the _-o_ flag 

`python windspeed.py test/test.png -o some/dir`

The default output is a csv file, `wind_result.csv`, containing wind intensity information for each image at _path_:

```
test.png, weak
```

Additionally, one may want to display intermediate steps, namely the visualization of step 1 and step 2 above. In case of _test.png_ for example, one would get

**Step 1**

![](data\flag_test2.png)



**Step 2**

![](data\wind_test2.png)



# Advanced

By default, a docker container is isolated from the host system, however it may be handy to exchange data with the image, be this input for the utility, or to retrieve its output. To this end, one can mount a volume in Linux as

`docker run -v /host/path:/docker/path -it wind bash`

One note, these have to be absolute paths. For example, assuming you cloned the repo in `~`, you could run

```
cd ~
mkdir out
docker run -v ~/windspeed_CLI/out:/app/out -it wind bash
```

For free, you can use the utilities' outputs (stored by default in _./out_) within the host system. Additionally, you can provide input to the utility by placing them in _out_.

For Windows the command is similar, however an intermediate step must be [done]([Docker on Windows — Mounting Host Directories | by Romin Irani | Romin Irani's Blog](https://rominirani.com/docker-on-windows-mounting-host-directories-d96f3f056a2c)).

