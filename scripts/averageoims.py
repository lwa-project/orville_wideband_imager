#!/usr/bin/env python
from lsl.common.mcs import mjdmpm_to_datetime
from lsl.common.paths import DATA as dataPath
from lsl.common.stations import lwasv
from lsl.misc import parser as aph
from scipy.interpolate import interp1d
from astropy.io import fits as astrofits
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.wcs import WCS
from astropy.time import Time
from datetime import datetime, timedelta
import os
import sys
import numpy
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import OrvilleImageDB
badfreqs = numpy.array([20.500,21.000,21.200,21.300,21.500,22.300,23.200,24.500,24.900,28.000,28.400,29.600,29.700,32.500,35.100])
parser = argparse.ArgumentParser()
parser.add_argument('filename',type=str, nargs='+')
args = parser.parse_args()
station = lwasv
for filename in args.filename:
    db = OrvilleImageDB.OrvilleImageDB(filename, 'r')
    outname = filename.replace(".oims","-avg.oims")
    ints = db.nint 
    nchan = db.header.nchan # number of frequency channels
    ngrid = db.header.ngrid # image size
    if nchan > 6: # Data needs averaging
        outname = filename.replace(".oims","-avg.oims")
        if os.path.isfile(outname):
            raise FileExistsError
        newdb = OrvilleImageDB.OrvilleImageDB(outname, mode='w', station=station.name)
        ints = db.nint 
        nchan = db.header.nchan # number of frequency channels
        ngrid = db.header.ngrid # image size
        data = numpy.zeros((ints,6,4,ngrid,ngrid))

        binchan = int(nchan/6)
        for i in range(ints):
            db.seek(i)
            hdr,alldata = db.read_image()
            oldbw = numpy.copy(hdr['bandwidth'])
            hdr['bandwidth'] = hdr['bandwidth']*binchan
            hdr['start_freq'] = hdr['start_freq'] - oldbw + (hdr['bandwidth']/2)
            for c1 in range(6):
                masked = 0
                for c2 in range(int(nchan/binchan)):
                    if round(c2*1e-6,2) in numpy.round(badfreqs,2):
                        masked +=1 
                    else:
                        data[i,c1]+= alldata[(c1*binchan)+c2]
                data[i,c1]/=(binchan-masked)
            newdb.add_image(hdr, data[i])
        newdb.close()
        db.close()
    else:
        print("already averaged")
        db.close()
