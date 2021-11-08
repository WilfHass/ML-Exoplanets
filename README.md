# ML-Exoplanets
University of Waterloo PHYS 449 Machine Learning project - recreating exoplanet detection paper.

The [GitHub](https://github.com/cshallue/exoplanet-ml) used to create the original paper.

The [README](https://github.com/cshallue/exoplanet-ml/tree/master/exoplanet-ml/astronet) for the astronet section explains how to  get and process the data. 

## Getting the Data
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
