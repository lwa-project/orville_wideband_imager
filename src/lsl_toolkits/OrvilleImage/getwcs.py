import numpy as np 
from astropy.wcs importWCS
from astropy.time import Time
import astropy.units as u

def getSVwcs(header, imSize):
    w = WCS(naxis=2)
    w.wcs.crpix = [imSize/2 + 0.5 * ((imSize+1)%2),imSize/2  + 0.5 * ((imSize+1)%2)]
    # 130 degrees is what is visible to the dipoles
    w.wcs.cdelt = numpy.array([130/imSize,130/imSize]) 
    HA, Dec = 357.38856977271047, 33.507121493107995
    LST = header['lst']
    theta_c = (89.17548407142988)*np.pi/180
    phi_c = (55.292881400599846)*np.pi/180
    xi = np.sin(phi_c)/np.tan(theta_c)
    eta = -np.cos(phi_c)/np.tan(theta_c)
    w.wcs.set_pv([(2,1,xi),(2,2,eta)])
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.crval = [Angle(LST-HA, u.deg).wrap_at(360*u.deg).degree, Dec]
    return w

def getGENERICwcs(header, imSize):
    w = WCS(naxis=2)
    w.wcs.crpix = [imSize/2 + 0.5 * ((imSize+1)%2),imSize/2  + 0.5 * ((imSize+1)%2)]
    # 130 degrees is what is visible to the dipoles
    w.wcs.cdelt = numpy.array([130/imSize,130/imSize]) 
    w.wcs.crval = [header['center_ra'],header['center_dec']]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    return w
