#!/bin/bash

# Where the data are stored
ORVILLE_DATA_DIR=/data/orville

# Where the software is stored
ORVILLE_SOFTWARE_DIR=/home/jdowell/orville3

# Where the configuration is stored
ORVILLE_CONFIG_DIR=`realpath $0 | xargs dirname`

# Parse the config. file
LWATV_CHAN=`grep lwatv_channel ${ORVILLE_CONFIG_DIR}/defaults.json | tail -n1 | sed -e 's/.*:[ \t]*//;s/,[ \t]*$//;s/"//g;'`

temp=`mktemp -d /tmp/health_check.XXXXXX`

# Find the latest CorrMatrix file and analyze it
cd ${temp}
latest_mjd=`ls -td ${ORVILLE_DATA_DIR}/matrices/* | head -n1`
latest_mat=`ls -td ${latest_mjd}/CorrMatrix_*_*000[0-4].npz | head -n1`
python3 ${ORVILLE_SOFTWARE_DIR}/analyze_correlations.py -s -w ${latest_mat}

# Fix up the filenames and upload the results
cd Summaries
ls * | sed -e 'p;s/_.*\./\./g' | xargs -n2 mv
mv Summary.txt advanced_health_check.txt
mv figure.png advanced_health_check.png
cd ..
rsync -e ssh -avH ./Summaries/* mcsdr@lwalab.phys.unm.edu:/var/www/${LWATV_CHAN}/

# Cleanup
cd /tmp
rm -rf ${temp}
