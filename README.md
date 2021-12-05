# ML-Exoplanets
University of Waterloo PHYS 449 Machine Learning project - recreating exoplanet detection paper.

The [GitHub](https://github.com/cshallue/exoplanet-ml) used to create the original paper.

The [README](https://github.com/cshallue/exoplanet-ml/tree/master/exoplanet-ml/astronet) for the astronet section explains how to  get and process the data. 

# PHYS449 ML Project - CNN

## Dependencies

- json
- numpy
- torch
- os
- random
- matplotlib
- argparse

## Running `main.py`

To run `main.py` with any variation:

```sh
python main.py --input [data-path] --network [cnn | fc | linear] --view [local | global | both] --user [user] --param [network]_[view].json --result results/ -v [0 | 1]
```

**Remember to set the user *everytime* you pull the repo** 

Typical usage for testing: 
```sh
python main.py --network [cnn | fc | linear] --view [local | global | both]
```

## Inputs
Takes as an input the data format in the 'running' section.
The torch tensor from real data, once it is obtained,
should be in /data and called in main.py

## Outputs
Running trains the neural network (depending on the input "view").
When finished training, it will output performance metrics
such as accuracy, precision, recall and AUC on the command line.
In addition, training and testing data is saved to a TensorBoard writer,
and can be viewed on TensorBoard.

Also outputs a csv file with the IDs of the light
curves inserted along with their probability between
0 and 1 of being a planet. The training and validation
data should also include the correct labels for each
light curve (planet/other).