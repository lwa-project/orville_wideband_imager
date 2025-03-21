import numpy as np 

import astropy.units as u
from astropy.wcs import WCS as AstroWCS
from astropy.time import Time as AstroTime
from astropy.coordinates import Angle as AstroAngle, EarthLocation, HADec, FK5


class WCS(AstroWCS):
    @classmethod
    def from_orville_header(kls, hdr):
        """
        Given an Orville imager header returned by `OrvilleImageDB.read_image()`,
        build a astropy.wcs.WCS object that represents all four axes in the image.
        """
        
        # Observation time
        t_obs = AstroTime(hdr['start_time'], format='mjd', scale='utc') # Time of observation
        
        # First, start off with a generic four axis WCS - RA, dec, Stokes, and frequency
        w = kls(naxis=4)
        w.wcs.radesys = 'FK5'
        w.wcs.specsys = 'TOPOCENT'
        w.wcs.equinox = t_obs.jyear
        w.wcs.crpix = [hdr['ngrid']/2 + 0.5 * ((hdr['ngrid']+1)%2),
                       hdr['ngrid']/2 + 0.5 * ((hdr['ngrid']+1)%2),
                       1,
                       1]
        w.wcs.cdelt = [-130/hdr['ngrid'],   # 130 degrees is what is visible to the dipoles
                        130/hdr['ngrid'],
                       1 if hdr['stokes_params'] == b'I,Q,U,V' else -1, # Not strictly correct since FITS has XX,YY,XY,YX
                       hdr['bandwidth']] 
        w.wcs.crval = [hdr['center_ra'],
                       hdr['center_dec'],
                       1 if hdr['stokes_params'] == b'I,Q,U,V' else -5,
                       hdr['start_freq']]
        w.wcs.ctype = ['RA---SIN',
                       'DEC--SIN',
                       'STOKES',
                       'FREQ']
        
        # Fix up the RA/DEC portions for Sevilleta
        if hdr['station'] == b'LWASV':
            ## Optimized phase center from Orville
            HA = 357.38856977271047
            Dec = 33.507121493107995 
            
            ## Assume the HA/Dec are measured in the epoch-of-date
            SV = EarthLocation(lat=34.348358*u.deg, lon=-106.885783*u.deg, height=1477.8*u.m)
            hc = HADec(HA*u.deg, Dec*u.deg, location=SV, obstime=t_obs)
            ec = hc.transform_to(FK5(equinox=t_obs))
            
            ## Adjust the CRVAL values for RA and dec
            w.wcs.crval[0] = ec.ra.deg
            w.wcs.crval[1] = ec.dec.deg
            
            ## Adjust the SIN projection center
            theta_c = (89.17548407142988)*np.pi/180
            phi_c = (55.292881400599846)*np.pi/180
            xi = np.sin(phi_c)/np.tan(theta_c)
            eta = -np.cos(phi_c)/np.tan(theta_c)
            w.wcs.set_pv([(2,1,xi),(2,2,eta)])
            
            ## Adjust the LONPOL value
            w.wcs.lonpole = 179.21441725378727
            
        return w


def getSVwcs(header, imSize):
    w = AstroWCS(naxis=2)
    w.wcs.crpix = [imSize/2 + 0.5 * ((imSize+1)%2),imSize/2  + 0.5 * ((imSize+1)%2)]
    # 130 degrees is what is visible to the dipoles
    w.wcs.cdelt = np.array([-130/imSize,130/imSize]) 
    HA = 357.38856977271047
    Dec = 33.507121493107995 
    
    SV = EarthLocation(lat=34.348358*u.deg, lon=-106.885783*u.deg, height=1477.8*u.m)
    t_obs = AstroTime(header['start_time'], format='mjd',location=SV) #time of observation
    
    # assume the HA/Dec are measured in the epoch-of-date
    LST = t_obs.sidereal_time("apparent").degree
    RA = AstroAngle(LST-HA, u.deg).wrap_at(360*u.deg).degree

    w.wcs.crval = [RA, Dec]

    theta_c = (89.17548407142988)*np.pi/180
    phi_c = (55.292881400599846)*np.pi/180
    xi = np.sin(phi_c)/np.tan(theta_c)
    eta = -np.cos(phi_c)/np.tan(theta_c)
    w.wcs.set_pv([(2,1,xi),(2,2,eta)])
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.lonpole = 179.21441725378727
    return w

def getGENERICwcs(header, imSize):
    w = AstroWCS(naxis=2)
    w.wcs.crpix = [imSize/2 + 0.5 * ((imSize+1)%2),imSize/2  + 0.5 * ((imSize+1)%2)]
    # 130 degrees is what is visible to the dipoles
    w.wcs.cdelt = np.array([-130/imSize,130/imSize]) 
    w.wcs.crval = [header['center_ra'],header['center_dec']]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    return w
