# Overview

This repo focuses on the effect that distributed model training has on training time. We will experiment with different number of nodes (up to 3) as well as with different types of neural network models (e.g. simple dense nn, ResNet 50, Bert).

# Prerequisites

You will need to create the conda enviroment from the enviroment.yml file.

```sh
cd distributed_training
conda env create -f environment.yml
conda activate distributed_training
```

# Usage

You will need to set up the different machines that the training will be distributed in by customasing the TF_CONFIG variables in the bash scripts in scripts files. 
The index should be different for each worker's script.
You will also need to change the path to the repository and the path to the conda environment as well according to your path. Please apply these path changes in the 
worker file in order to save your results.
Start the distributed training by running the following command:

```sh
/path/to/repo/scripts/run_distributed_training.sh -n nodes -d dataset
```

The nodes argument can be from 1 up to 3 (for up to 3 different machines).

The supported dataset arguments are the following:

* cifar_10
* bert_movies
* natural_images_densenet
* fashion_mnist
* mnist

# Contributors

*   Kontaras Marinos el17050
*   Bouras Dimitrios-Stamatios el17072
*   Siozos Theodoros el17083
