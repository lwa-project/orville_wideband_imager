import OrvilleImageDB
from lsl.common.mcs import mjdmpm_to_datetime
from astropy.io import fits as astrofits
from datetime import datetime, timedelta
import os
import sys
import numpy
import argparse

def main(args):
    for filename in args.filename:
        db = OrvilleImageDB.OrvilleImageDB(filename, 'r')
        
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
        
        for i,(hdr,alldata) in enumerate(db):
            if args.verbose:
                print("  working on integration #%i" % (i+1))
            ## Reverse the axis order so we can get it right in the FITS file
            for chan, stokesdata in enumerate(alldata.data):
                for data,stokes in  zip(stokesdata,['I','Q','U','V']):
                    ## Save the image size for later
                    imSize = data.shape[-1]
                    
        
                    ## Zero outside of the horizon so avoid problems
                    pScale = psize
                    sRad   = 360.0/pScale/numpy.pi / 2
                    x = numpy.arange(data.shape[-2]) - 0.5
                    y = numpy.arange(data.shape[-1]) - 0.5
                    x,y = numpy.meshgrid(x,y)
                    invalid = numpy.where( ((x-imSize/2.0)**2 + (y-imSize/2.0)**2) > (sRad**2) )
                    data[invalid[0], invalid[1]] = 0.0
                    ext = imSize/(2*sRad)
        
                    ## Convert the start MJD into a datetime instance and then use
                    ## that to come up with a stop time
                    mjd = int(hdr['start_time'])
                    mpm = int((hdr['start_time'] - mjd)*86400.0*1000.0)
                    tInt = hdr['int_len']*86400.0
                    dateObs = mjdmpm_to_datetime(mjd, mpm)
                    dateEnd = dateObs + timedelta(seconds=int(tInt), microseconds=int((tInt-int(tInt))*1000000))
                    if args.verbose:
                        print("    start time: %s" % dateObs)
                        print("    end time: %s" % dateEnd)
                        print("    integration time: %.3f s" % tInt)
                        print("    frequency: %.3f MHz" % header['freq'])
        
                    ## Create the FITS HDU and fill in the header information
                    hdu = astrofits.PrimaryHDU(data=data)
                    hdu.header['TELESCOP'] = 'LWA1'
                    ### Date and time
                    hdu.header['DATE-OBS'] = dateObs.strftime("%Y-%m-%dT%H:%M:%S")
                    hdu.header['END_UTC'] = dateEnd.strftime("%Y-%m-%dT%H:%M:%S")
                    hdu.header['EXPTIME'] = tInt
                    ### Coordinates - sky
                    hdu.header['CTYPE1'] = 'RA---SIN'
                    hdu.header['CRPIX1'] = imSize/2 + 1 + 0.5 * ((imSize+1)%2)
                    hdu.header['CDELT1'] = -360.0/(2*sRad)/numpy.pi
                    hdu.header['CRVAL1'] = hdr['center_ra']
                    hdu.header['CUNIT1'] = 'deg'
                    hdu.header['CTYPE2'] = 'DEC--SIN'
                    hdu.header['CRPIX2'] = imSize/2 + 1 + 0.5 * ((imSize+1)%2)
                    hdu.header['CDELT2'] = 360.0/(2*sRad)/numpy.pi
                    hdu.header['CRVAL2'] = hdr['center_dec']
                    hdu.header['CUNIT2'] = 'deg'
                    ### Coordinates - Stokes parameters
                    hdu.header['CTYPE3'] = 'STOKES'
                    hdu.header['CRPIX3'] = 1
                    hdu.header['CDELT3'] = 1
                    hdu.header['CRVAL3'] = 1
                    hdu.header['LONPOLE'] = 180.0
                    hdu.header['LATPOLE'] = 90.0
                    ### LWA1 approximate beam size
                    midfreq = (hdr['start_freq']  + ((chan+1)*hdr['bandwidth']/2))
                    beamSize = 2.2*74e6/midfreq
                    hdu.header['BMAJ'] = beamSize/psize
                    hdu.header['BMIN'] = beamSize/psize
                    hdu.header['BPA'] = 0.0
                    ### Frequency
                    hdu.header['RESTFREQ'] = midfreq
        
                    ## Write it to disk
                    outName = "oims_%.3fMHz_%s_stokes%s.fits" % (midfreq/1e6, dateObs.strftime("%Y-%m-%dT%H-%M-%S"),stokes)
                    hdulist = astrofits.HDUList([hdu,])
                    hdulist.writeto(outName, overwrite=args.force)
        
        
        db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='convert the images contained in one or more .oims files into FITS images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str, nargs='+',
                        help='filename to convert')
    parser.add_argument('-f', '--force', action='store_true',
                        help='force overwriting of FITS files')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='be verbose during the conversion')
    args = parser.parse_args()
    main(args)
