#!/usr/bin/env python
from lsl.common.mcs import mjdmpm_to_datetime
from lsl.common.paths import DATA as dataPath
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
def calcbeamprops(az,alt,header,freq):

    # az and alt need to be the same shape as the image we will correct

    i = 0
    beamDict = numpy.load(os.path.join(dataPath, 'lwa1-dipole-emp.npz'))
    polarpatterns = []
    for beamCoeff in (beamDict['fitX'], beamDict['fitY']):
        alphaE = numpy.polyval(beamCoeff[0,0,:],freq )
        betaE =  numpy.polyval(beamCoeff[0,1,:],freq )
        gammaE = numpy.polyval(beamCoeff[0,2,:],freq )
        deltaE = numpy.polyval(beamCoeff[0,3,:],freq )
        alphaH = numpy.polyval(beamCoeff[1,0,:],freq )
        betaH =  numpy.polyval(beamCoeff[1,1,:],freq )
        gammaH = numpy.polyval(beamCoeff[1,2,:],freq )
        deltaH = numpy.polyval(beamCoeff[1,3,:],freq )
        corrFnc = None

        def compute_beam_pattern(az, alt, corr=corrFnc):
            zaR = numpy.pi/2 - alt*numpy.pi / 180.0
            azR = az*numpy.pi / 180.0

            c = 1.0
            if corrFnc is not None:
                c = corrFnc(alt*numpy.pi / 180.0)
                c = numpy.where(numpy.isfinite(c), c, 1.0)

            pE = (1-(2*zaR/numpy.pi)**alphaE)*numpy.cos(zaR)**betaE + gammaE*(2*zaR/numpy.pi)*numpy.cos(zaR)**deltaE
            pH = (1-(2*zaR/numpy.pi)**alphaH)*numpy.cos(zaR)**betaH + gammaH*(2*zaR/numpy.pi)*numpy.cos(zaR)**deltaH

            return c*numpy.sqrt((pE*numpy.cos(azR))**2 + (pH*numpy.sin(azR))**2)
        # Calculate the beam
        pattern = compute_beam_pattern(az, alt)
        polarpatterns.append(pattern)
        i += 1
    beamDict.close()
    return polarpatterns[0], polarpatterns[1]

def pbcorroims(header,imSize,chan):
    mjd = int(header['start_time'])
    mpm = int((header['start_time'] - mjd)*86400.0*1000.0)
    tInt = header['int_len']*86400.0
    dateObs = mjdmpm_to_datetime(mjd, mpm)
    x = numpy.arange(imSize) - 0.5
    y = numpy.arange(imSize) - 0.5
    x,y = numpy.meshgrid(x,y)
    pScale = header['pixel_size']
    sRad   = 360.0/pScale/numpy.pi / 2
    crval1 = header['center_ra']*numpy.pi/180
    crpix1 = imSize/2 + 1 + 0.5 * ((imSize+1)%2) 
    cdelt1 = numpy.pi*(-360.0/(2*sRad)/numpy.pi)/180
    crval2 = header['center_dec']*numpy.pi/180
    crpix2 = imSize/2 + 1 + 0.5 * ((imSize+1)%2) 
    cdelt2 = numpy.pi*(360.0/(2*sRad)/numpy.pi)/180
    ra = ((crval1 + (x - crpix1)*cdelt1/(numpy.cos(crval2)))*180/numpy.pi) 
    dec = (crval2 + cdelt2*(y-crpix2))*180/numpy.pi
    # Make dec go between -90 and 90
    # Adjust RA accordingly
    decover = dec>90
    decdiff = dec[decover] - 90
    dec[decover] = dec[decover] - decdiff
    ra[decover] +=180
    decoverneg = dec<-90
    decdiffneg = dec[decoverneg] + 90
    dec[decoverneg] = dec[decoverneg] + decdiffneg
    ra[decoverneg] +=180
    ra = ra % 360
    
    sc = SkyCoord(ra,dec,unit='deg')
    lwasv = EarthLocation.from_geodetic(-106.885664,34.348562, height=1475) 
    time = Time(dateObs.strftime("%Y-%m-%dT%H:%M:%S"),format="isot")
    aa = AltAz(location=lwasv, obstime=time)
    myaltaz = sc.transform_to(aa)
    
    alt = myaltaz.alt.deg
    az = myaltaz.az.deg
    # Keep alt between 0 and 90, adjust az accordingly
    negalt = alt < 0
    alt[negalt] += 90
    az[negalt] + 180
    freq = (int(header['start_freq'])  + ((int(chan)+1)*int(header['bandwidth'])/2))
    XX,YY = calcbeamprops(az,alt,header,freq)
    return XX,YY
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
        hdrlist = []
        data = numpy.zeros((ints,nchan,4,ngrid,ngrid))
        for i in range(ints):
            db.seek(i)
            hdr,alldata = db.read_image()
            hdrlist.append(hdr)
            data[i] = numpy.asarray(alldata.data)
        hdr = hdrlist[0]
        if args.diff:
            tmpdata = numpy.copy(data)
            data = numpy.zeros((ints-1,6,4,ngrid,ngrid))
            for i in range(ints-1):
                data[i] = tmpdata[i+1] - tmpdata[i]
        for chan in range(nchan):
            hdulist = astrofits.HDUList()
            for myint in range(len(data)):
                hdr = hdrlist[myint]

                imdata = data[myint,chan,:,:,:]
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
                    XX,YY = pbcorroims(hdrlist[myint],imSize,chan)
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
                hdu.header['DATE-OBS'] = dateObs.strftime("%Y-%m-%dT%H:%M:%S")
                hdu.header['END_UTC'] = dateEnd.strftime("%Y-%m-%dT%H:%M:%S")
                hdu.header['EXPTIME'] = tInt
                ### LWA1 approximate beam size
                midfreq = (hdr['start_freq']  + ((chan)*hdr['bandwidth']) + (hdr['bandwidth']/2))
                beamSize = 2.2*74e6/midfreq
                hdu.header['BMAJ'] = beamSize/psize
                hdu.header['BMIN'] = beamSize/psize
                hdu.header['BPA'] = 0.0
                ### Frequency
                hdu.header['RESTFREQ'] = midfreq
                
                ## Write it to disk
                hdulist.append(hdu)
            filedir,filebase = os.path.split(os.path.abspath(os.path.expanduser(filename)))
            if args.diff:
                outName = filedir + filebase[0:13] + f"{round(midfreq*1e-6,1)}MHz-diff.fits"
            else: 
                outName = filedir + filebase[0:13] + f"{round(midfreq*1e-6,1)}MHz.fits"
            hdulist.writeto(outName, overwrite=args.force)
        
        
        db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='convert the images contained in one or more .oims files into FITS images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str, nargs='+',
                        help='filename to convert')
    parser.add_argument('-d', '--diff', action='store_true',
                        help='Generate diff images')
    parser.add_argument('-f', '--force', action='store_true',
                        help='force overwriting of FITS files')
    parser.add_argument('-p', '--pbcorr', action='store_true',
                        help='Perform primary beam correction on Stokes I')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='be verbose during the conversion')
    args = parser.parse_args()
    main(args)
