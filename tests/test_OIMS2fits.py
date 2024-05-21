"""
Unit test for OrvilleImageDB module.
"""

import os
import sys
import glob
import numpy
import tempfile
import unittest
import subprocess

from lsl_toolkits.OrvilleImager import OrvilleImageDB
from astropy.io import fits

currentDir = os.path.abspath(os.getcwd())
if os.path.exists(os.path.join(currentDir, 'tests', 'test_OIMS2fits.py')):
    MODULE_BUILD = currentDir
else:
    MODULE_BUILD = None
    
run_scripts_tests = False
if MODULE_BUILD is not None:
    run_scripts_tests = True

__version__  = "0.3"
__author__    = "Jayce Dowell"


oimsFile = os.path.join(os.path.dirname(__file__), 'data', 'test.oims')

@unittest.skipUnless(run_scripts_tests, "cannot determine correct script path to use")
class OIMS2fits_tests(unittest.TestCase):
    """A unittest.TestCase collection of unit tests for the OrvilleImageDB
    module."""
    
    testPath = None

    def setUp(self):
        """Turn off all numpy warnings and create the temporary file directory."""

        numpy.seterr(all='ignore')
        self.testPath = tempfile.mkdtemp(prefix='test-OIMS2fits-', suffix='.tmp')
    def test_OIMS2fits_run(self):
        """Create fits from oims"""
        
        fitsFile = glob.glob(oimsFile.replace(".oims", "*.fits"))
        if fitsFile:
            for f in fitsFile:
                try:
                    os.remove(f)
                except OSError:
                    pass
                    
        with open('OIMS2fits.log', 'w') as logfile:
            try:
                cmd = [sys.executable, 'scripts/OIMS2fits.py', oimsFile]
                status = subprocess.check_call(cmd, stdout=logfile)
            except subprocess.CalledProcessError:
                status = 1
                
        if status == 1:
            with open('OIMS2fits.log', 'r') as logfile:
                print(logfile.read())
        os.unlink('OIMS2fits.log')
        self.assertEqual(status, 0)
        
        fitsFile = glob.glob(oimsFile.replace(".oims", "*.fits"))
        for f in fitsFile:
            with fits.open(f) as hdul:
                nchan = len(fitsFile)
                ints = len(hdul)
                stokes = hdul[0].data.shape[0]
                xdata = hdul[0].data.shape[1]
                ydata = hdul[0].data.shape[2]

            with OrvilleImageDB(oimsFile, 'r') as db:
                self.assertEqual(nchan, db.header.nchan)
                self.assertEqual(ints, db.nint)
                self.assertEqual(stokes, len(db.header.stokes_params.split(b',')))
                self.assertEqual(xdata, db.header.ngrid)
                self.assertEqual(ydata, db.header.ngrid)
                
            try:
                os.remove(f)
            except OSError:
                pass


class OIMS2fits_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which contains all of the OrvilleImageDB units 
    tests."""
    
    def __init__(self):
        unittest.TestSuite.__init__(self)
        
        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(OIMS2fits_tests)) 


if __name__ == '__main__':
    unittest.main()
