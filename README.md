# DragPoser: Motion Reconstruction from Variable Sparse Tracking Signals via Latent Space Optimization

:warning: Under construction... :construction: :construction:

## Getting Started

1. Clone the repository onto your local system.
2. Navigate to the `python` directory.
3. Create a virtual environment with: ``python -m venv env`` (tested on Python 3.9).
4. Activate the created virtual environment.
5. Install the necessary packages from the requirements file with: ``pip install -r requirements.txt``.
6. Download and install [PyTorch](https://pytorch.org/get-started/locally/).

### Evaluate

**One BVH file**
```bash
python .\src\eval_drag.py .\models\model_dancedb .\data\example\eval\xample.bvh --config .\config\6_trackers_config.json
```

**A directory with BVH files**
```bash
python .\src\eval_drag.py .\models\model_dancedb .\data\example\eval\ --config .\config\6_trackers_config.json
```

Results will be saved in ``.\data\``

### Train

Training and evaluation data should be in a directory similar to ``.\data\example\``
Inside that directory there must be two files (eval and train) with the .bvh files for training.
Note that the included ``.\data\example\`` does not have enough data for training, it only includes an example file from the preprocessed AMASS dataset.


**1. Train the VAE**
```bash
python .\src\train.py .\data\example\ name --fk
```

**2. Train the temporal predictor**
```bash
python .\src\train_temporal.py .\data\example\ name
```

## License

- The code in this repository is released under the MIT license. Please, see the [LICENSE](LICENSE) for further details.

- The model weights in this repository and the data are licensed under CC BY-SA 4.0 license as found in the [LICENSE_data](LICENSE_data) file.
