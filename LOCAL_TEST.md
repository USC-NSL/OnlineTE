# Testing `OnlineTE` On A Single Machine

> **Note:** This _WILL_ be slow, and will not reflect the performance correctly, but it _MUST_ show the effects of warm-starting and quick convergence after the first traffic matrix correctly.

# Setting Up The Environment

## With Docker
A Docker file, `Dockerfile` has been provided. Please install Docker on your machine. Navigate to the root of the repository and then:
```
sudo docker build -t onlinete .
```
To build the image. Then:
```
sudo docker run -it onlinete /bin/bash
```
To drop into a ready environment. You may now proceed to run tests as all dependencies are already there.

## With Anaconda
An Anaconda ENV file, `environment.yml` has been provided if you wish to set up a local environment.
```
conda env create -f environment.yml
```

## Locally
If you have your own virtual environment or just do now care, then:
```
sudo apt-get update
python -m pip install uv
python -m uv pip install -r requirements.txt
```
But this may break some of your existing packages.

## Compiling Protocol Buffers and Unzipping The Zoo

> **If you are using Docker, this is already done for you. Feel free to skip!**

We use topologies from the [Internet Topology Zoo](topology-zoo.org). The website appears unavailable in many regions, thus a copy of the
dataset exists within this repository. You can read the paper about this wonderful dataset below:
```bib
@article{article,
author = {Knight, Simon and Nguyen, Hung and Falkner, Nickolas and Bowden, Rhys and Roughan, Matthew},
year = {2011},
month = {11},
pages = {1765-1775},
title = {The Internet Topology Zoo},
volume = {29},
journal = {IEEE Journal on Selected Areas in Communications},
doi = {10.1109/JSAC.2011.111002}
}
```
```
# Unzip the topology zoo dataset
unzip ./topologies/zoo.zip
```

We use `gRPC` for most of our communication backend (we are not using `Ray`, as `OnlineTE` was never intended to use shared memory and must run distributed over switches in the network). The protocol buffers are under `protos` and can be compiled with:
```
python protos/__init__.py
```
## (Optional) Gurobi License
We use PDLP from [Google or-tools](https://github.com/google/or-tools) within `OnlineTE`, but Gurobi is still much more accurate and provides baselines for comparison.

Some of our tests are large and requrie a licensed model. Gurobi provides academic licenses for free as long as an academic email is available. Once such a license exists, please put it in your home directory and you can now use our `centralized` solvers to check the accuracy of `OnlineTE`.

----
> **Now, please proceed to evaluation as described in `EVALUATION.md`**
----