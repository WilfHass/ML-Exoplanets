# ML-Exoplanets
University of Waterloo PHYS 449 Machine Learning project - recreating exoplanet detection paper.

The [GitHub](https://github.com/cshallue/exoplanet-ml) used to create the original paper.

The [paper](https://arxiv.org/abs/1712.05044) where architectures and results are recreated

The [README](https://github.com/cshallue/exoplanet-ml/tree/master/exoplanet-ml/astronet) for the astronet section explains how to get and process the data. 

# PHYS449 ML Project

## Dependencies

- json
- numpy
- torch
- os
- random
- matplotlib
- argparse
- sys
- sklearn
- datetime

## Running `main.py`

To run `main.py` with any combination of network and view:

```sh
python main.py --input [data-path] --network [cnn | fc | linear] --view [local | global | both] --user [user] --param [network]_[view].json --result results/ -v [0 | 1]
```

Typical usage for testing: 
```sh
python main.py --network [cnn | fc | linear] --view [local | global | both] --user [s | d | w | a]
```

## Inputs
- Torch tensor data, converted from TFRecords and stored in ```/data/torch_data_ID/```
- JSON file with hyperparameters, stored in ```/param/```

## Outputs
Input data is used to train the neural network chosen by command line arguments.
At the end of training, the hyperparameters and performance metrics are saved in TensorBoard format,
where they can be later accessed.

In the verbose mode, performance metrics are output to console, such as:
- Training loss
- Test loss
- Test accuracy
- Test AUC
- 5 best and 5 worst predicted TCEs
  - Kepler ID
  - TCE planet number
  - label
  - prediction
  - difference

Heatmaps and precision-recall plots are generated and saved in ```/results/plots/```

CSV and pt (PyTorch tensor) files with Kepler IDs, planet numbers, labels and predictions in ```/results/testing_probabilities/```

## Performance metrics
To view the performance metrics after a run:
```sh
tensorboard --logdir=results/TensorBoard
```

To view the parameters of a specific architecture (ex cnn):

```sh
tensorboard --logdir=results/TensorBoard/<architecture>/<view>
```