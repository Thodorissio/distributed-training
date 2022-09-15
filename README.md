# Important Note to contributors

O,ti kainourio package katevazete sto conda enviroment valte to sto additional_dependencies.txt ws eksis:

```sh
pip install package==version
```

Episis link gia to report:
https://www.overleaf.com/7682747425xzwvhrzyythq

**Simiosi apo boura** (tha diagrafei sto telos apto readme ) : exo to arxiko report se odt kai docx giati dn ksero latex .mporeite na sublirosete ekei kai na ta kanoume latex sto telos i kati tetoio. (tha diagrafoun ta report arxeia apo to github repo sto telos)

# Overview

This repo focuses on the effect that distributed model training has on training time. We will experiment with different number of nodes (up to 3) as well as with different types of neural network models (e.g. simple dense nn, ResNet 50, Bert).

# Prerequisites

You will need to create the conda enviroment from the enviroment.yml file.

```sh
cd distributed_training
conda env create -f environment.yml
conda activate distributed_training
```

# TODO

*   Run and save the experiments
*   Create report
*   Upload results on README and polish repo

# Contributors

*   Kontaras Marinos el17050
*   Bouras Dimitrios-Stamatios el17072
*   Siozos Theodoros el17083
