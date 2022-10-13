# ELo-SAC: Evolving Losses + Soft Actor Critic

This repository is the official implementation of ELo-SACv2 as a part of our paper **Does Self-supervised Learning Really Improve Reinforcement Learning from Pixels?** ([openreview](https://openreview.net/forum?id=fVslVNBfjd8), [arxiv](https://arxiv.org/abs/2206.05266)) at NeurIPS 2022.

Our implementation is based on [SAC+AE](https://github.com/denisyarats/pytorch_sac_ae) by Denis Yarats and [CURL](https://github.com/MishaLaskin/curl) by Michael Laskin. 

You may also want to check ELo-SACv3 at the main branch of this repository, and Atari experiments were done in a separate codebase (Check [ELo-Rainbow](https://github.com/LostXine/elo-rainbow)).

## Installation 

All of the dependencies are in the `conda_env.yml` file. They can be installed manually or with the following command:

```
conda env create -f conda_env.yml
```

Change the server IP and port in the `search-server-ip` file if necessary.

## Instructions

First, start the search server using `bash start-server.sh` or the following command:

```
python3 search-server.py --port 61888 --timeout 24
```

which will start a HTTP server listening at the given port. 
The server runs a PSO (Particle Swarm Optimization) algorithm and distributes tasks to the clients with GPUs.
Timeout means how many hours the server will wait for the client to report results before it assigns the same task to another client.
Our optimization status is stored at `save_server/evol_rl.npy` and will be automatically loaded.
One could start a new search by assigning `--path` to a new file.

To start the parameter search on clients, run `bash search.sh`. 
The client will connect to the HTTP server and request the hyper-parameters for training.
When the training completes, the client will report the evaluation results to the server and requests a new task.

Run `bash check_status.sh` or `bash check_full_status.sh` to check the search status.

To stop the search, **stop** the current server and **restart** the search server with `--stop True` (see the `server-stop.sh` file).
All the clients will stop searching after finishing the current search.

To evaluate the optimal combination, run `bash eval-s09.sh` and it will start to train ELo-SAC agents in 6 DMControl environments with 10 random seeds.

See the `train.py` file for training hyper-parameters.

## Contact

1. Issue
2. email: xiangli8@cs.stonybrook.edu