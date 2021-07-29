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
@conference{numeroso2021,
      title={MEG: Generating Molecular Counterfactual Explanations for Deep Graph Networks},
      author={Danilo Numeroso and Davide Bacciu},
      year={2021},
      date={2021-07-18},
      booktitle={Proceedings of the International Joint Conference on Neural Networks (IJCNN 2021)},
      organization={IEEE},
      keywords={deep learning for graphs, explainable AI, graph data, structured data processing},
      pubstate={forthcoming},
      tppubtype={conference}
}
```
