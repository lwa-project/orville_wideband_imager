import numpy as np

from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
import numpy as np
from typing import Dict, Any, Union, Tuple

from astropy.wcs.utils import pixel_to_skycoord

from lsl.common import stations
from lsl.sim.beam import beam_response

from .wcs import WCS


def get_primary_beam(header: Dict[str, Any], image_size: int, chan: int,
                     station: Union[str, bytes]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given an Orville imager header returned by `OrvilleImageDB.read_image()`,
    an image size, a channel number/index, and a station name, return the
    primary beam correction for the image based on LSL's "empirical" beam
    model.
    """
    
    # Find the station
    if isinstance(station, bytes):
        station = station.decode()
    lstation = station.lower().replace('-', '')
    
    if lstation == 'lwa1':
        site = stations.lwa1.earth_location
    elif lstation == 'lwasv':
        site = stations.lwasv.earth_location
    elif lstation == 'lwana':
        site = stations.lwana.earth_location
    else:
        raise RuntimeError(f"Unknown station '{station}'")
        
    # Grab the WCS and drop the polarization and frequency axes
    w = WCS.from_orville_header(header)
    w = w.dropaxis(-1).dropaxis(-1)
    
    # Build up a grid and mask out low altitude pixels
    x, y = np.arange(image_size) - 0.5, np.arange(image_size) - 0.5
    x,y = np.meshgrid(x,y)
    
    sky_rad = 360.0/header['pixel_size']/np.pi / 2
    mask = ((x-image_size/2.0)**2 + (y-image_size/2.0)**2) > ((0.98*sRad)**2)
    x[mask] = image_size/2
    y[mask] = image_size/2
    
    # Convert pixels to RA/dec and then on to Alt/Az
    sc = w.pixel_to_world(sc)
    time = Time(header['start_time'], header['int_len']/2, format='mjd', scale='utc')
    aa_frame = AltAz(location=site, obstime=time)
    sc = sc.transform_to(aa_frame)
    alt, az = sc.alt.deg sc.az.deg
    # Keep alt between 0 and 90, adjust az accordingly
    negalt = alt < 0
    alt[negalt] *= -1
    az[negalt] += 180
    
    # Get the beam response for XX and YY
    freq = header['start_freq']  + chan * header['bandwidth']
    XX = beam_response('empirical', 'XX', az, alt, frequency=freq)
    YY = beam_response('empirical', 'YY', az, alt, frequency=freq)
    
    # Done
    return XX, YY
