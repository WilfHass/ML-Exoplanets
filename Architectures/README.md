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

To run `main.py` with the two variations shown:

Local view of data: 
```sh
python main.py --input data-path --view local --param [nn]_local.json -v [verbosity (0 or 1)]
```
Global view of data:
```sh
python main.py --input data-path --view global --param [nn]_global.json -v [verbosity (0 or 1)]
```
Both views of data:
```sh
python main.py --input data-path --view both --param [nn]_both.json -v [verbosity (0 or 1)]
```
where [nn] should be put in as linear, fc or cnn, depending on which architecture is being used.

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