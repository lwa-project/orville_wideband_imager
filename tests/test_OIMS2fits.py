"""
Unit test for OIMS2fits.py script.
"""

import os
import sys
import glob
import numpy as np
import shutil
import tempfile
import unittest
import subprocess

from lsl_toolkits.OrvilleImage import OrvilleImageHDF5
from lsl_toolkits.OrvilleImage.legacy import OrvilleImageDB
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
o5File = os.path.join(os.path.dirname(__file__), 'data', 'test.o5')


@unittest.skipUnless(run_scripts_tests, "cannot determine correct script path to use")
class OIMS2fits_tests(unittest.TestCase):
    """A unittest.TestCase collection of unit tests for the OIMS2fits.py
    script."""
    
    testPath = None

    def setUp(self):
        """Turn off all numpy warnings and create the temporary file directory."""

        np.seterr(all='ignore')
        self.testPath = tempfile.mkdtemp(prefix='test-OIMS2fits-', suffix='.tmp')
        
    def tearDown(self):
        try:
            shutil.rmtree(self.testPath)
        except OSError:
            pass
            
    def test_OIMS2fits_oims_run(self):
        """Create fits from oims"""
        
        with open(os.path.join(self.testPath, 'OIMS2fits.log'), 'w') as logfile:
            try:
                cmd = [sys.executable, os.path.join(MODULE_BUILD, 'scripts', 'OIMS2fits.py'), oimsFile]
                status = subprocess.check_call(cmd, stdout=logfile, cwd=self.testPath)
            except subprocess.CalledProcessError:
                status = 1
                
        if status == 1:
            with open(os.path.join(self.testPath, 'OIMS2fits.log'), 'r') as logfile:
                print(logfile.read())
        self.assertEqual(status, 0)
        
        fitsFile = glob.glob(os.path.join(self.testPath, "*.fits"))
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
                
    def test_OIMS2fits_o5_run(self):
        """Create fits from o5"""
        
        with open(os.path.join(self.testPath, 'OIMS2fits.log'), 'w') as logfile:
            try:
                cmd = [sys.executable, os.path.join(MODULE_BUILD, 'scripts', 'OIMS2fits.py'), o5File]
                status = subprocess.check_call(cmd, stdout=logfile, cwd=self.testPath)
            except subprocess.CalledProcessError:
                status = 1
                
        if status == 1:
            with open(os.path.join(self.testPath, 'OIMS2fits.log'), 'r') as logfile:
                print(logfile.read())
        self.assertEqual(status, 0)
        
        fitsFile = glob.glob(os.path.join(self.testPath, "*.fits"))
        for f in fitsFile:
            with fits.open(f) as hdul:
                nchan = len(fitsFile)
                ints = len(hdul)
                stokes = hdul[0].data.shape[0]
                xdata = hdul[0].data.shape[1]
                ydata = hdul[0].data.shape[2]

            with OrvilleImageHDF5(o5File, 'r') as db:
                self.assertEqual(nchan, db.header.nchan)
                self.assertEqual(ints, db.nint)
                self.assertEqual(stokes, len(db.header.stokes_params.split(b',')))
                self.assertEqual(xdata, db.header.ngrid)
                self.assertEqual(ydata, db.header.ngrid)


class OIMS2fits_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which contains all of the OIMS2fits.py
    unit tests."""
    
    def __init__(self):
        unittest.TestSuite.__init__(self)
        
        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(OIMS2fits_tests)) 


if __name__ == '__main__':
    unittest.main()
