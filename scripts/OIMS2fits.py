#!/usr/bin/env python3

from lsl.common.mcs import mjdmpm_to_datetime
from astropy.io import fits as astrofits
from datetime import datetime, timedelta
import os
import sys
import numpy as np
import argparse

from lsl_toolkits.OrvilleImage import OrvilleImageHDF5
from lsl_toolkits.OrvilleImage.legacy import OrvilleImageDB

from lsl_toolkits.OrvilleImage.wcs import WCS
from lsl_toolkits.OrvilleImage.utils import get_primary_beam


def main(args):
    for filename in args.filename:
        OrvilleReader = OrvilleImageHDF5
        if os.path.splitext(filename)[1] == '.oims':
            OrvilleReader = OrvilleImageDB
            
        with OrvilleReader(filename, 'r') as db:
        
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
            # Collect header and data from the whole file
            hdrlist = []
            if args.index is not None:
                if not args.diff:
                    data = np.zeros((1,nchan,4,ngrid,ngrid), dtype=np.float32)
                    try:
                        hdr,alldata = db.read_image(args.index)
                    except TypeError:
                        db.seek(args.index)
                        hdr,alldata = db.read_image()
                    hdrlist.append(hdr)
                    data[0] = np.asarray(alldata.data)
                else:
                    data = np.zeros((1,nchan,4,ngrid,ngrid), dtype=np.float32)
                    # FIRST do the next image
                    try:
                        hdr,alldata = db.read_image(args.index+1)
                    except TypeError:
                        db.seek(args.index+1)
                        hdr,alldata = db.read_image()
                    data[0] = np.asarray(alldata.data)
                    # Next subtract our image
                    try:
                        hdr,alldata = db.read_image(args.index)
                    except TypeError:
                        db.seek(args.index)
                        hdr,alldata = db.read_image()
                    hdrlist.append(hdr)
                    data[0] = data[0] - np.asarray(alldata.data)
                hdr = hdrlist[0]
            else:    
                data = np.zeros((ints,nchan,4,ngrid,ngrid), dtype=np.float32)
                for i in range(ints):
                    try:
                        hdr,alldata = db.read_image(i)
                    except TypeError:
                        db.seek(i)
                        hdr,alldata = db.read_image()
                    hdrlist.append(hdr)
                    data[i] = np.asarray(alldata.data)
                hdr = hdrlist[0]
                if args.diff:
                    tmpdata = np.copy(data)
                    data = np.zeros((len(data)-1,nchan,4,ngrid,ngrid), dtype=np.float32)
                    for i in range(len(data)):
                        data[i] = tmpdata[i+1] - tmpdata[i]
            for chan in range(nchan):
                if args.channel is not None:
                    if chan!=args.channel:
                        continue
                hdulist = astrofits.HDUList()
                for myint in range(len(data)):
                    hdr = hdrlist[myint]
                    if args.diff:
                        imdata = args.corrfac*data[myint,chan,:,:,:]
                    else:
                        imdata = args.corrfac*(data[myint,chan,:,:,:] - args.background)
                    imSize = ngrid    
                    
                    ## Zero outside of the horizon so avoid problems
                    pScale = psize
                    sRad   = 360.0/pScale/np.pi / 2
                    x = np.arange(data.shape[-2]) - 0.5
                    y = np.arange(data.shape[-1]) - 0.5
                    x,y = np.meshgrid(x,y)
                    invalid = np.where( ((x-imSize/2.0)**2 + (y-imSize/2.0)**2) > (sRad**2) )
                    imdata[:,invalid[0], invalid[1]] = 0.0
                    ext = imSize/(2*sRad)
                    if args.pbcorr:
                        XX,YY = get_primary_beam(hdrlist[myint], imSize, chan, station)
                        imdata[0]/=((XX+YY)/2)
                    
                    ## Convert the start MJD into a datetime instance and then use
                    ## that to come up with a stop time
                    mjd = int(hdr['start_time'])
                    mpm = int((hdr['start_time'] - mjd)*86400.0*1000.0)
                    # determine if header is in seconds or days
                    if round(hdr['int_len']) > 0:
                        tInt = hdr['int_len']
                    else:
                        tInt = hdr['int_len']*86400.0
                    dateObs = mjdmpm_to_datetime(mjd, mpm)
                    dateEnd = dateObs + timedelta(seconds=int(tInt), microseconds=int((tInt-int(tInt))*1000000))
                    if args.verbose:
                        print("    start time: %s" % dateObs)
                        print("    end time: %s" % dateEnd)
                        print("    integration time: %.3f s" % tInt)
                        print("    frequency: %.3f MHz" % header['freq'])
                        
                    ## Create the FITS HDU and fill in the header information
                    hdu = astrofits.ImageHDU(data=imdata)
                    if isinstance(station, bytes):
                        station = station.decode()
                    hdu.header['TELESCOP'] = station
                    hdu.header['EXPTIME'] = tInt
                    ### Coordinates
                    w = WCS.from_orville_header(hdr)
                    w = w.dropaxis(-1)  # Trim off the FREQ axis
                    wcs_hdr = w.to_header()
                    for key in wcs_hdr:
                        hdu.header[key] = wcs_hdr[key]
                    hdu.header['DATE-OBS'] = dateObs.strftime("%Y-%m-%dT%H:%M:%S")
                    hdu.header['END_UTC'] = dateEnd.strftime("%Y-%m-%dT%H:%M:%S")
                    hdu.header['EXPTIME'] = tInt
                    ### LWA1 approximate beam size
                    midfreq = (hdr['start_freq']  + ((chan)*hdr['bandwidth']) )
                    beamSize = 2.2*74e6/midfreq
                    hdu.header['BMAJ'] = beamSize/psize
                    hdu.header['BMIN'] = beamSize/psize
                    hdu.header['BPA'] = 0.0
                    ### Frequency
                    hdu.header['RESTFREQ'] = midfreq
                    hdu.header['RESTFRQ'] = midfreq
                    hdu.header['RESTBW'] = hdr['bandwidth']
                    hdu.header['SPECSYS'] = 'TOPOCENT'
                    
                    ## Write it to disk
                    hdulist.append(hdu)
                filedir,filebase = os.path.split(os.path.abspath(os.path.expanduser(filename)))
                if args.output_dir is not None:
                    filedir = args.output_dir
                    
                if args.diff:
                    outName = filedir + os.path.sep + filebase[0:13] + f"{round(midfreq*1e-6,1)}MHz-diff.fits"
                else: 
                    outName = filedir + os.path.sep + filebase[0:13] + f"{round(midfreq*1e-6,1)}MHz.fits"
                if args.index is not None:
                    outName = outName.replace(".fits",f"-{args.index}.fits")
                hdulist.writeto(outName, overwrite=args.force)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='convert the images contained in one or more .oims files into FITS images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str, nargs='+',
                        help='filename to convert')
    parser.add_argument('-b', '--background',type=float,
                        default = 0,
                        help='Uncorrected background flux')
    parser.add_argument('-c', '--corrfac', type=float,
                        default=1,
                        help="Flux correction multiplicative factor")
    parser.add_argument('--channel', type=int,
                        help="Only image this channel")
    parser.add_argument('-d', '--diff', action='store_true',
                        help='Generate diff images')
    parser.add_argument('-f', '--force', action='store_true',
                        help='force overwriting of FITS files')
    parser.add_argument('-i', '--index', type=int,
                        help='Only output this index')
    parser.add_argument('-p', '--pbcorr', action='store_true',
                        help='Perform primary beam correction on Stokes I')
    parser.add_argument('-o', '--output-dir',
                        help='directory to write FITS files to (default: same directory as the .oims files)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='be verbose during the conversion')
    args = parser.parse_args()
    main(args)
