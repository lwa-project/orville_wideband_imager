import numpy as np 
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import Angle, EarthLocation, SkyCoord
import astropy.units as u

def getSVwcs(header, imSize):
    w = WCS(naxis=2)
    w.wcs.crpix = [imSize/2 + 0.5 * ((imSize+1)%2),imSize/2  + 0.5 * ((imSize+1)%2)]
    # 130 degrees is what is visible to the dipoles
    w.wcs.cdelt = np.array([130/imSize,130/imSize]) 
    HA = 357.38856977271047*u.deg
    Dec = 33.507121493107995*u.deg 
    
    SV = EarthLocation(lat=34.348358*u.deg, lon=-106.885783*u.deg, height=1477.8*u.m)
    t_obs = Time(header['start_time'], format='mjd') #time of observation
    
    # assume the HA/Dec are measured in the epoch-of-date
    hadec = SkyCoord(ha=HA, dec=Dec, frame='hadec', obstime=t_obs, location=SV)
    radec = hadec.transform_to('fk5')
    w.wcs.crval = [radec.ra.deg, radec.dec.deg]

    theta_c = (89.17548407142988)*np.pi/180
    phi_c = (55.292881400599846)*np.pi/180
    xi = np.sin(phi_c)/np.tan(theta_c)
    eta = -np.cos(phi_c)/np.tan(theta_c)
    w.wcs.set_pv([(2,1,xi),(2,2,eta)])
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.lonpole = 179.21441725378727
    return w

def getGENERICwcs(header, imSize):
    w = WCS(naxis=2)
    w.wcs.crpix = [imSize/2 + 0.5 * ((imSize+1)%2),imSize/2  + 0.5 * ((imSize+1)%2)]
    # 130 degrees is what is visible to the dipoles
    w.wcs.cdelt = np.array([130/imSize,130/imSize]) 
    w.wcs.crval = [header['center_ra'],header['center_dec']]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    return w
