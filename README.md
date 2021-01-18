# PIDNet
Deep Learning based 3D Particle Identification

Code, dataset, and trained models for the paper "PIDNet: Deep Learning based $\bold{e^-/\pi^0}$ separation in the Single Phase Liquid Argon TPC"

If you find this code useful in you work, please cite:
```
@article{bay2020pidnet,
  title={PIDNet: Deep Learning based $\bold{e^-/\pi^0}$ separation in the Single Phase Liquid Argon TPC},
  author={},
  journal={arXiv preprint arXiv:},
  year={2020}
}
```

#### Particle Topologies

<img src='imgs/particle-topologies.png' width=512/> 


## Getting Started

### Installation

- Clone this repo:

```bash
cd ~
git clone https://https://github.com/yasinalm/PIDNet
cd PIDNet
```

### Prerequisites

- Linux (Tested on Ubuntu 16.04)
- NVIDIA GPU (Tested on Nvidia GTX 1080 Ti)
- CUDA, CuDNN
- Python 3
- tensorflow-gpu>=2.0.0
- Tensorboard>=2.0.0
- scipy

- Install [Tensorflow](http://tensorflow.org) 2.0.0+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can use an installation script `./scripts/conda_deps.sh`. Alternatively, you can create a new Conda environment using `conda env create -f environment.yml`.

### Dataset

- Our dataset consists of particles generated in Monte Carlo simulations.
- The data split we used can be downloaded [here](https://1drv.ms/u/s!AsXONMc_kIHJb1pqU_1CGv9RBXk?e=5xGbvI).

### Dataset Organization

Data needs to be arranged in the following format:

```python
.
├── dataset5d
│   └── 1500sp_evts
│       ├── electron-38323
│       ├── muon-62190
│       ├── pionminus-39144
│       ├── pionzero-35674
│       ├── proton-36793
│       ├── test_files35k_5p.csv
│       └── train_files35k_5p.csv
└── PIDNet
    ├── figs
    │   └── confusion_matrix_pointnet.png
    ├── logs
    │   └── checkpoints
    │       ├── checkpoint
    │       ├── iter-102828.data-00000-of-00002
    │       ├── iter-102828.data-00001-of-00002
    │       └── iter-102828.index
    ├── src
    │   ├── dataset_utils.py
    │   ├── eval_test_set.py
    │   ├── eval_test_set_tsne.ipynb
    │   ├── inference.py
    │   ├── match_pred_momentum.ipynb
    │   ├── model.py
    │   ├── plot_pred_momentum.ipynb
    │   ├── roc_curve.ipynb
    │   ├── train.py
    └── utils
        ├── helpers.py
        └── visualize.py


```

#### Network Architecture

<img src='imgs/architecture.png' width=1024/>

### Steps to reproduce the results

1. Prepapare the data using the command: 'prepare_data/python prepare_fixedsp_data.py'
2. Create a list of train and test files with the Jupyter notebook file prepare_data/generate_train_test_files.ipynb
    1. Just sequentially run the code blocks in the notebook.
3. Run the model in the pid_net folder: 'python src/train.py'

### Training

 To train a model:

```bash
python -u src/train.py |& tee logs/log_training_$(date '+%Y%m%d-%H%M%S').txt
```

To train with custom parameters:
```bash
python -u src/train.py --n_classes=5 --batch_size=64 --epochs=50  |& tee logs/log_training_$(date '+%Y%m%d-%H%M%S').txt
```

To resume training from a checkpoint (if directory, uses the latest checkpoint in the directory):
```bash
python -u src/train.py --init_weight=${ckpt_dir}  |& tee logs/log_training_$(date '+%Y%m%d-%H%M%S').txt
```

If you have multiple GPUs on your machine, you can also run the multi-GPU version training:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -u src/train.py |& tee logs/log_training_$(date '+%Y%m%d-%H%M%S').txt
```

- To see more intermediate results, check out  `./logs/checkpoints/`.
- To view training results and loss plots, run `python -m visdom.server` and click the URL <http://localhost:8097.>

### Pre-trained Models

- The pretrained model is saved at `./logs/checkpoints/`.


### Testing

To test the model:

```bash
python -u src/eval_test_set.py --init_weight=${ckpt_dir} |& tee logs/log_test.txt
```

- The statistics will be prompted to screen and confusion matrix result will be saved to a png file here: `./figs/confusion_matrix.png`.

### Confusion Matrix, ROC Curve, and T-SNE Visualization
<p float="left">
<img src='imgs/confusion_matrix.png' width=300/>
<img src='imgs/roc.png' width=300/>
<img src='imgs/tsne.png' width=300/>
</p>
  
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


## Reference

If you find our work useful in your research please consider citing our paper:

```
@article{bay2020pidnet,
  title={PIDNet: Deep Learning based $\bold{e^-/\pi^0}$ separation in the Single Phase Liquid Argon TPC},
  author={},
  journal={arXiv preprint arXiv:},
  year={2020}
}
```
