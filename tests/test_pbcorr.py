"""
Unit test for OrvilleImage.utils module.
"""

import os
import sys
import glob
import numpy as np
import tempfile
import unittest
import subprocess
import shutil

from lsl_toolkits.OrvilleImage.legacy import OrvilleImageDB
from lsl_toolkits.OrvilleImage.utils import get_primary_beam
from astropy.io import fits

currentDir = os.path.abspath(os.getcwd())
if os.path.exists(os.path.join(currentDir, 'tests', 'test_OIMS2fits.py')):
    MODULE_BUILD = currentDir
else:
    MODULE_BUILD = None
    
run_scripts_tests = False
if MODULE_BUILD is not None:
    run_scripts_tests = True

__version__  = "0.1"
__author__    = "Jayce Dowell"


oimsFile = os.path.join(os.path.dirname(__file__), 'data', 'test.oims')


@unittest.skipUnless(run_scripts_tests, "cannot determine correct script path to use")
class pbcorr_tests(unittest.TestCase):
    """A unittest.TestCase collection of unit tests for the OrvilleImage.utils
    module."""
    
    testPath = None

    def setUp(self):
        """Turn off all numpy warnings and create the temporary file directory."""

        np.seterr(all='ignore')
        self.testPath = tempfile.mkdtemp(prefix='test-OIMS2fits-', suffix='.tmp')
        
    
    def tearDown(self):
        """Remove the test path directory and its contents"""
        
        shutil.rmtree(self.testPath, ignore_errors=True)
        self.testPath = None
        
    def test_pbcor_run(self):
        """Test running utils.get_primary_beam"""

        hdr = {'station': 'LWASV',
               'stokes_params': 'I,Q,U,V',
               'ngrid': 144,
               'pixel_size': 0.9027777777777778,
               'start_time': 59430.09473379629,
               'int_len': 5.787037037037037e-05,
               'fill': 0.0,
               'lst': 0.6669193610247549,
               'start_freq': 38700000.0,
               'stop_freq': 55200000.0,
               'bandwidth': 3300000.0,
               'weighting': 'natural',
               'center_ra': 242.0902199689118,
               'center_dec': 33.3576388888889,
               'center_az': 90.97437629624612,
               'center_alt': 88.06662829704536,
               'asp_filter': -1,
               'asp_atten_1': -1,
               'asp_atten_2': -1,
               'asp_atten_s': -1
              }
        ngrid = hdr['ngrid']
        pScale = hdr['pixel_size']
        XX, YY = get_primary_beam(hdr, ngrid, 0, 'LWASV')
        x = np.arange(ngrid)
        y = np.arange(ngrid)
        x ,y = np.meshgrid(x,y)
        invbeam = 1/((XX+YY)/2)
        sRad = 360.0/pScale/np.pi/2
        rms = []
        for i in range(1,10):
            selpix  = (((x-ngrid/2.0)**2 + (y-ngrid/2.0)**2) > ((((i-1)/10)*sRad)**2)) & (((x-ngrid/2.0)**2 + (y-ngrid/2.0)**2) < (((i/10)*sRad)**2))
            rms.append(np.sqrt(np.average(invbeam[selpix]**2)))
        for r1,r2 in zip(rms[:-1],rms[1:]):
            self.assertTrue(r2>r1)


class pbcorr_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which contains all of the OrvilleImage.utils
    units tests."""
    
    def __init__(self):
        unittest.TestSuite.__init__(self)
        
        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(pbcorr_tests)) 


if __name__ == '__main__':
    unittest.main()
