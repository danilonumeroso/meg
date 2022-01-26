# MEG: Molecular Explanation Generator
This repository contains the implementation of [MEG](https://arxiv.org/abs/2104.08060) (IJCNN 2021).

# Usage
We assume miniconda (or anaconda) to be installed.

### Install dependencies
Run the following commands: 
```
source setup/install.sh [cpu | cu92 | cu101 | cu102]
conda activate meg
```

### Train DGN

Train the DGN to be explained by running:
```
python train_dgn.py [tox21 | esol] <experiment_name>
```

### Generate counterfactuals

To generate counterfactual explanations for a specific sample, run:
```
python train_meg.py [tox21 | esol] <experiment_name> --sample <INTEGER>
```
Results will be saved at ```runs/<dataset_name>/<experiment_name>/meg_output```.

# Bibtex
```
@inproceedings{numeroso2021,
      author={Numeroso, Danilo and Bacciu, Davide},
      booktitle={2021 International Joint Conference on Neural Networks (IJCNN)}, 
      title={MEG: Generating Molecular Counterfactual Explanations for Deep Graph Networks}, 
      year={2021},
      volume={},
      number={},
      pages={1-8},
      doi={10.1109/IJCNN52387.2021.9534266}
}
```
