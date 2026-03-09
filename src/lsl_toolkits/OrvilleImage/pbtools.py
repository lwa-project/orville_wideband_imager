from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from lsl.sim.beam import beam_response
from lsl_toolkits.OrvilleImage.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
import numpy as np

def calcbeamprops(az,alt,header,freq):

    # az and alt need to be the same shape as the image we will correct
    
    polarpatterns = []
    polarpatterns.append(beam_response('empirical', 'XX', az, alt, frequency=freq))
    polarpatterns.append(beam_response('empirical', 'YY', az, alt, frequency=freq))
    
    return polarpatterns[0], polarpatterns[1]

def pbcorroims(header,imSize,chan,station):
    pScale = header['pixel_size']
    sRad   = 360.0/pScale/np.pi / 2
    w = WCS.from_orville_header(header)
    w = w.dropaxis(-1).dropaxis(-1)
    x = np.arange(imSize) - 0.5
    y = np.arange(imSize) - 0.5
    x,y = np.meshgrid(x,y)
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
    freq = int(header['start_freq'])  + (chan*header['bandwidth'])
    XX,YY = calcbeamprops(az,alt,header,freq)
    return XX,YY
