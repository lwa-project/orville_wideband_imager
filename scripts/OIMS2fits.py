#!/usr/bin/env python3

from lsl.common.mcs import mjdmpm_to_datetime
from lsl.sim.beam import beam_response
from lsl.misc import parser as aph
from scipy.interpolate import interp1d
from astropy.io import fits as astrofits
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.wcs.utils import pixel_to_skycoord
from astropy.time import Time
from datetime import datetime, timedelta
import os
import sys
import numpy
import argparse

from lsl_toolkits.OrvilleImage import OrvilleImageDB
from lsl_toolkits.OrvilleImage.wcs import WCS


def calcbeamprops(az,alt,header,freq):

    # az and alt need to be the same shape as the image we will correct
    
    polarpatterns = []
    polarpatterns.append(beam_response('empirical', 'XX', az, alt, frequency=freq))
    polarpatterns.append(beam_response('empirical', 'YY', az, alt, frequency=freq))
    
    return polarpatterns[0], polarpatterns[1]

def pbcorroims(header,imSize,chan,station):
    pScale = header['pixel_size']
    sRad   = 360.0/pScale/numpy.pi / 2
    w = WCS.from_orville_header(header)
    w = w.dropaxis(-1).dropaxis(-1)
    x = numpy.arange(imSize) - 0.5
    y = numpy.arange(imSize) - 0.5
    x,y = numpy.meshgrid(x,y)
    maskpix  = ((x-imSize/2.0)**2 + (y-imSize/2.0)**2) > ((0.98*sRad)**2)
    x[maskpix] = imSize/2
    y[maskpix] = imSize/2
    sc = pixel_to_skycoord(x, y, wcs=w, mode='wcs')
    # Need date and location for converting to altaz
    if station == b'LWASV':
        site = EarthLocation.from_geodetic(-106.885783, 34.348358, height=1477.8) 
    elif station == b'LWANA':
        site = EarthLocation.from_geodetic(-107.640, 34.247, height=2134)
    time = Time(header['start_time'], header['int_len']/2, format='mjd', scale='utc')
    aa = AltAz(location=site, obstime=time)
    myaltaz = sc.transform_to(aa)
    alt = myaltaz.alt.deg
    az = myaltaz.az.deg
    # Keep alt between 0 and 90, adjust az accordingly
    negalt = alt < 0
    alt[negalt] *= -1
    az[negalt] += 180
    freq = (int(header['start_freq'])  + (chan*header['bandwidth'])
    XX,YY = calcbeamprops(az,alt,header,freq)
    return XX,YY

def main(args):
    for filename in args.filename:
        with OrvilleImageDB(filename, 'r') as db:
        
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
                    data = numpy.zeros((1,nchan,4,ngrid,ngrid))
                    db.seek(args.index)
                    hdr,alldata = db.read_image()
                    hdrlist.append(hdr)
                    data[0] = numpy.asarray(alldata.data)
                else:
                    data = numpy.zeros((1,nchan,4,ngrid,ngrid))
                    # FIRST do the next image
                    db.seek(args.index+1)
                    hdr,alldata = db.read_image()
                    data[0] = numpy.asarray(alldata.data)
                    # Next subtract our image
                    db.seek(args.index)
                    hdr,alldata = db.read_image()
                    hdrlist.append(hdr)
                    data[0] = data[0] - numpy.asarray(alldata.data)
                hdr = hdrlist[0]
            else:    
                data = numpy.zeros((ints,nchan,4,ngrid,ngrid))
                for i in range(ints):
                    db.seek(i)
                    hdr,alldata = db.read_image()
                    hdrlist.append(hdr)
                    data[i] = numpy.asarray(alldata.data)
                hdr = hdrlist[0]
                if args.diff:
                    tmpdata = numpy.copy(data)
                    data = numpy.zeros((len(data)-1,nchan,4,ngrid,ngrid))
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
                    sRad   = 360.0/pScale/numpy.pi / 2
                    x = numpy.arange(data.shape[-2]) - 0.5
                    y = numpy.arange(data.shape[-1]) - 0.5
                    x,y = numpy.meshgrid(x,y)
                    invalid = numpy.where( ((x-imSize/2.0)**2 + (y-imSize/2.0)**2) > (sRad**2) )
                    imdata[:,invalid[0], invalid[1]] = 0.0
                    ext = imSize/(2*sRad)
                    if args.pbcorr:
                        XX,YY = pbcorroims(hdrlist[myint],imSize,chan,station)
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
                    hdu.header['TELESCOP'] = station.decode()
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
