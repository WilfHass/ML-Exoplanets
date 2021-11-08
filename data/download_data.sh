# Download compressed archive (~105.7MB). DOESN'T WORK
# This method currently does not work.
wget https://storage.googleapis.com/kepler-ml/astronet/data/kepler/kepler-tce-tfrecords-20180220.tar.gz

# Create directory for the extracted TFRecord files.
BASE_DIR="${HOME}/astronet/"
mkdir -p ${BASE_DIR}

# Extract files.
tar -xvf kepler-tce-tfrecords-20180220.tar.gz -C ${BASE_DIR}

# Extracted files are located in the 'tfrecord' subdirectory.
TFRECORD_DIR="${BASE_DIR}/tfrecord"
ls -l ${TFRECORD_DIR}

# Clean up archive file.
rm kepler-tce-tfrecords-20180220.tar.gz