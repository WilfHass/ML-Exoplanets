# PHYS449 ML Project - CNN
Local and global data only.
The CNN is currently in scaffolding mode, and needs
to be filled in.

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

Local data: 
```sh
python main.py --param param/param_local.json --res-path outputs
```
Global data:
```sh
python main.py --param param/param_global.json --res-path outputs
```

## Inputs
Takes as an input the data format in the 'running' section.
The torch tensor from real data, once it is obtained,
should be in /data and called in main.py

## Outputs
The CNN outputs a txt file with performance metrics
such as accuracy, AUC, recall.

Also outputs a csv file with the IDs of the light
curves inserted along with their probability between
0 and 1 of being a planet. The training and validation
data should also include the correct labels for each
light curve (planet/other).

## Working together
- Every time you see PLEASE that means I didn't code in 
that part. 
- The scaffolding is mostly place holders for code to make
it work. 
- If you don't understand what I'm specifying
just let me know