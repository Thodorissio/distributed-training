# Important Note to contributors

O,ti kainourio package katevazete sto conda enviroment valte to sto additional_dependencies.txt ws eksis:

```sh
pip install package==version
```

# Overview

This repo focuses on the effect that distributed model training has on training time. We will experiment with different number of nodes (up to 3) as well as with different types of neural network models (e.g. simple dense nn, ResNet 50, Bert).

# Prerequisites

You will need to create the conda enviroment from the enviroment.yml file.

```sh
conda env create -f environment.yml
```

# TODO

*   Add multinode setup script
*   Improve model examples
*   Run multinode training with 2 and 3 different nodes
*   Create report
*   Upload results on README and polish repo

# Contributors

*   Kontaras Marinos el17050
*   Mpouras Dimitrios-Stamatios el17072
*   Siozos Theodoros el17083