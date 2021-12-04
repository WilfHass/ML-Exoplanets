## PHYS 449 ML project:
## Convert TFRecords used in paper to Torch format

## to understand TFRecords, read:
# https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c

import numpy as np
import sys
import argparse
import tensorflow as tf
import torch

sys.path.append('TFTensors')

parser = argparse.ArgumentParser(description='Convert TFTensor to numpy array')
parser.add_argument('--input', default="TFTensors", help='path to TFTensors directory')
parser.add_argument('--output', default="TorchTensors_ID", type=str, help="path to TorchTensors directory")
args = parser.parse_args()

input_loc = str(args.input)
output_loc = str(args.output)

filename_in = args.input + "/val-00000-of-00001"
filename_out = args.output + "/val-00000-of-00001_ID"



## Do not add the .tfrecord extension at the end of the filenames! They have no extension
dataset = tf.data.TFRecordDataset(filename_in)

# description of the features. Found out the hard way.
feature_description = {
    'tce_plnt_num': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'tce_max_mult_ev': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'tce_time0bk': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'tce_depth': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'tce_prad': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'local_view': tf.io.FixedLenFeature([201], tf.float32, default_value=np.zeros(201, dtype=np.float32)),
    'av_training_set': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'tce_duration': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'tce_impact': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'spline_bkspace': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'tce_model_snr': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'kepid': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'tce_period': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'av_pred_class': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'global_view': tf.io.FixedLenFeature([2001], tf.float32, default_value=np.zeros(2001, dtype=np.float32)),
}

def parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

parsed_dataset = dataset.map(parse_function)
print(len(list(parsed_dataset)))

num_samples = len(list(parsed_dataset))

array = []
for parsed_record in parsed_dataset.take(num_samples):
    local_list = list(parsed_record['local_view'].numpy().flatten())
    global_list = list(parsed_record['global_view'].numpy().flatten())
    label = parsed_record['av_training_set'].numpy()
    kepid = parsed_record['kepid'].numpy()
    tce_plnt_num = parsed_record['tce_plnt_num'].numpy()

    if label == b'PC':
        label = float(1.0)
    else:
        label = float(0.0)

    array.append(local_list + global_list + [label, kepid, tce_plnt_num])

print([label, kepid, tce_plnt_num])

np_array = np.array(array, dtype='float32')
tensor = torch.from_numpy(np_array)

print(tensor)
torch.save(tensor, filename_out)



































