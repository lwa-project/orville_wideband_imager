# Orville Wideband Imager

[![Test](https://github.com/lwa-project/orville_wideband_imager/actions/workflows/main.yml/badge.svg)](https://github.com/lwa-project/orville_wideband_imager/actions/workflows/main.yml)

[![Paper](https://img.shields.io/badge/arXiv-2103.03347-blue.svg)](https://arxiv.org/abs/2103.03347)   [![Memo](https://img.shields.io/badge/lwa%20memo-215-blue)](http://leo.phys.unm.edu/~lwa/memos/memo/lwa0215.pdf)


## Description
The Orville Wideband Imager is a realtime GPU-based all-sky imager for the output
of the Advanced Digitial Processor (ADP) broadband correlator that runs at LWA-SV.
Orville receives visibility data from ADP for 32,896 baselines, images the data,
and writes the images to the disk in a [binary frame-based format called "OIMS"](https://github.com/lwa-project/orville_wideband_imager/blob/main/src/lsl_toolkits/OrvilleImage/OrvilleImageDB.py).
The imaging is performed using a _w_-stacking algorithm for the non-coplanarity of
the array.  For each image, the sky is projected onto the two dimensional plane using
orthographic sine projection.  To reduce the number of _w_-planes needed during _w_-stacking,
the phase center is set to a location approximately 2 degrees off zenith that minimizes
the spread in the _w_ coordinate. The gridding operation is based on the Romein gridder
implemented as part of the [EPIC project](https://github.com/epic-astronomy/EPIC).  Every 5
seconds, the imager produces 4 Stokes (I, Q, U and V) images in 198 channels, each with
100 kHz bandwidth.

## Data Archive
Orville data with reduced spectral resolution (six 3.3 MHz channels) are available at the [LWA data archive](https://lda10g.alliance.unm.edu/Orville/).

## Reading OIMS Files
You can use the `OrvilleImageDB.py` Python module to read the data stored in an OIMS file:
```
from lsl_toolkits.OrvilleImager import OrvilleImageDB

db = OrvilleImageDB(oimsFile, 'r')

# Get parameters from the input file

ints = db.nint # number of integrations
station =  db.header.station # station info
stokes = db.header.stokes_params # Stokes parameter info
inp_flag = db.header.flags # flag info
file_start = db.header.start_time # file start time
file_end = db.header.stop_time   # file end time
ngrid = db.header.ngrid # image size (x-axis)
psize = db.header.pixel_size # angular size of a pixel (at zenith)
nchan = db.header.nchan # number of frequency channels

# There are two main ways to read data from the OIMS file:
#  - From a single integration
#  - From all integrations at once

# 1. Reading a particular image integration
#    (Usually 720 integrations (each 5 seconds) in an hour) 

hdr, dat = db.__getitem__(710) # collecting header and data from the 710 th integration

# The dat array contains the image data in a four dimensional array of the form [nchan,stokes,xgrid,ygrid]
# where nchan = 198 frequency channels, stokes = 4 [stokes (I, Q, U,V)], xgrid = grid size in the x direction,
# ygrid = grid size in the y direction (typically xgrid = ygrid = ngrid)
# Copy over to numpy arrays for further processing (e.g. image subtraction and transient searches).

# Individual header items can be accessed from the hdr dictionary:

t = hdr['start_time'] # starting time in MJD
int_len = hdr['int_len'] # length of each integration
lst = hdr['lst'] # starting LST time of observation
start_freq = hdr['start_freq']/1e+6 # starting Frequency in MHz
stop_freq = hdr['stop_freq']/1e+6  # stopping Frequency in MHz
bandwidth = hdr['bandwidth']/1e+6 # bandwidth of the observation
cent_ra = hdr['center_ra'] # phase center RA
cent_dec = hdr['center_dec'] # phase center Dec
cent_az = hdr['center_az']   # phase center coordinates, azimuth, Ideally towards zenith, changed due to w-projection 
cent_alt = hdr['center_alt'] # phase center coordinates, elevation, Ideally towards zenith, changed due to w-projection 


# 2. Reading all integrations at once

hdr_list, data_all = db.read_all()

# data_all contains the image data in a 5d masked array of the form [integration,nchan,stokes,xgrid,ygrid]
# where integration is the number of integrations (typically 720 per hour) and the other axes are the same
# as above. hdr_list is a list of dictionaries, one per integration. This can easily be converted to a 
# pandas dataframe object by doing:

import pandas as pd
hdr_df = pd.DataFrame(hdr_list)

# Close the database file
db.close()
```
