# Getting the Data
## To use Our Code
You need wget and tensorflow to run these scripts
### Changing size of dataset
```sh
python reduce_dataset.py --num [insert integer for dataset size]
```
For some reason, this isn't linear (when num is 500, there's 301 targets, num=100 45 targets, num=150 77 targets)

### Downloading Mikulski data
```
python process/generate_download_script.py --kepler_csv_file=dr24_tce.csv --download_dir=kepler/
./get_kepler.sh
```
Creates the fits files in the kepler/ directory


## README for The Paper's Code

### Download the Kepler Data
To create our training set, the first step is to download the list of labeled TCEs that will comprise the training set. You can download the DR24 TCE Table in CSV format from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=q1_q17_dr24_tce). Ensure the following columns are selected:

- `rowid`: Integer ID of the row in the TCE table.
- `kepid`: Kepler ID of the target star.
- `tce_plnt_num`: TCE number within the target star.
- `tce_period`: Period of the detected event, in days.
- `tce_time0bk`: The time corresponding to the center of the first detected event in Barycentric Julian Day (BJD) minus a constant offset of 2,454,833.0 days.
- `tce_duration`: Duration of the detected event, in hours.
- `av_training_set`: Autovetter training set label; one of PC (planet candidate), AFP (astrophysical false positive), NTP (non-transiting phenomenon), UNK (unknown).

Next, you will need to download the light curves of the stars corresponding to the TCEs in the training set. These are available at the Mikulski Archive for Space Telescopes. However, you almost certainly don't want all of the Kepler data, which consists of almost 3 million files, takes up over a terabyte of space, and may take several weeks to download! To train our model, we only need to download the subset of light curves that are associated with TCEs in the DR24 file. To download just those light curves, follow these steps:

**NOTE**: Even though we are only downloading a subset of the entire Kepler dataset, the files downloaded by the following script take up about 90 GB.

```sh
# Filename containing the CSV file of TCEs in the training set.
TCE_CSV_FILE="${HOME}/astronet/dr24_tce.csv"

# Directory to download Kepler light curves into.
KEPLER_DATA_DIR="${HOME}/astronet/kepler/"

# Generate a bash script that downloads the Kepler light curves in the training set.
python astronet/data/generate_download_script.py \
  --kepler_csv_file=${TCE_CSV_FILE} \
  --download_dir=${KEPLER_DATA_DIR}

# Run the download script to download Kepler light curves.
./get_kepler.sh
```
The final line should read: `Finished downloading 12669 Kepler targets to ${KEPLER_DATA_DIR}`


Let's explore the downloaded light curve of the Kepler-90 star! Note that Kepler light curves are divided into four quarters each year, which are separated by the quarterly rolls that the spacecraft made to reorient its solar panels. In the downloaded light curves, each .fits file corresponds to a specific Kepler quarter, but some quarters are divided into multiple `.fits` files.

### Process Kepler Data
The command below will generate a set of sharded TFRecord files for the TCEs in the training set. Each `tf.Example` proto will contain the following light curve representations:

- `global_view`: Vector of length 2001: a "global view" of the TCE.
- `local_view`: Vector of length 201: a "local view" of the TCE.

In addition, each `tf.Example` will contain the value of each column in the input TCE CSV file. The columns include:

- `rowid`: Integer ID of the row in the TCE table.
- `kepid`: Kepler ID of the target star.
- `tce_plnt_num`: TCE number within the target star.
- `av_training_set`: Autovetter training set label.
- `tce_period`: Period of the detected event, in days.

```sh
# Directory to save output TFRecord files into.
TFRECORD_DIR="${HOME}/astronet/tfrecord"

# Preprocess light curves into sharded TFRecord files using 5 worker processes.
bazel-bin/astronet/data/generate_input_records \
  --input_tce_csv_file=${TCE_CSV_FILE} \
  --kepler_data_dir=${KEPLER_DATA_DIR} \
  --output_dir=${TFRECORD_DIR} \
  --num_worker_processes=5
```

When the script finishes you will find 8 training files, 1 validation file and 1 test file in `TFRECORD_DIR`. The files will match the patterns `train-0000?-of-00008`, `val-00000-of-00001` and `test-00000-of-00001` respectively.

Explore the generated representations of Kepler-90g in the output:

```python
# Launch iPython (or Python) from the exoplanet-ml directory.
ipython

In[1]:
import matplotlib.pyplot as plt
import numpy as np
import os.path
import tensorflow as tf

In[2]:
KEPLER_ID = 11442793  # Kepler-90
TFRECORD_DIR = "/path/to/tfrecords/dir"

In[3]:
# Helper function to find the tf.Example corresponding to a particular TCE.
def find_tce(kepid, tce_plnt_num, filenames):
  for filename in filenames:
    for record in tf.python_io.tf_record_iterator(filename):
      ex = tf.train.Example.FromString(record)
      if (ex.features.feature["kepid"].int64_list.value[0] == kepid and
          ex.features.feature["tce_plnt_num"].int64_list.value[0] == tce_plnt_num):
        print("Found {}_{} in file {}".format(kepid, tce_plnt_num, filename))
        return ex
  raise ValueError("{}_{} not found in files: {}".format(kepid, tce_plnt_num, filenames))

In[4]:
# Find Kepler-90 g.
filenames = tf.gfile.Glob(os.path.join(TFRECORD_DIR, "*"))
assert filenames, "No files found in {}".format(TFRECORD_DIR)
ex = find_tce(KEPLER_ID, 1, filenames)

In[5]:
# Plot the global and local views.
global_view = np.array(ex.features.feature["global_view"].float_list.value)
local_view = np.array(ex.features.feature["local_view"].float_list.value)
fig, axes = plt.subplots(1, 2, figsize=(20, 6))
axes[0].plot(global_view, ".")
axes[1].plot(local_view, ".")
plt.show()
```
