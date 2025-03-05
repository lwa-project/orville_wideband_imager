"""
Unit test for OIMS2o5.py script.
"""

import os
import sys
import glob
import numpy
import shutil
import tempfile
import unittest
import subprocess

from lsl_toolkits.OrvilleImage import OrvilleImageHDF5
from lsl_toolkits.OrvilleImage.legacy import OrvilleImageDB
from astropy.io import fits

currentDir = os.path.abspath(os.getcwd())
if os.path.exists(os.path.join(currentDir, 'tests', 'test_OIMS2o5.py')):
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
class OIMS2o5_tests(unittest.TestCase):
    """A unittest.TestCase collection of unit tests for the OIMS2o5.py
    script."""
    
    testPath = None

    def setUp(self):
        """Turn off all numpy warnings and create the temporary file directory."""

        numpy.seterr(all='ignore')
        self.testPath = tempfile.mkdtemp(prefix='test-OIMS2o5-', suffix='.tmp')
        
    def tearDown(self):
        try:
            shutil.rmtree(self.testPath)
        except OSError:
            pass
            
    def test_OIMS2o5_run(self):
        """Create fits from oims"""
        
        fitsFile = glob.glob(oimsFile.replace(".oims", "*.o5"))
        if fitsFile:
            for f in fitsFile:
                try:
                    os.remove(f)
                except OSError:
                    pass
                    
        with open(os.path.join(self.testPath, 'OIMS2o5.log'), 'w') as logfile:
            try:
                cmd = [sys.executable, 'scripts/OIMS2o5.py', oimsFile]
                status = subprocess.check_call(cmd, stdout=logfile, cwd=self.testPath)
            except subprocess.CalledProcessError:
                status = 1
                
        if status == 1:
            with open(os.path.join(self.testPath, 'OIMS2o5.log'), 'r') as logfile:
                print(logfile.read())
        self.assertEqual(status, 0)
        
        o5File = glob.glob(os.path.join(self.testPath, os.path.basename(oimsFile).replace(".oims", "*.o5"))
        for f in o5File:
            with OrvilleImageHDF5(f, 'r') as o5:
                nchan = o5.header.nchan
                ints = o5.nint
                _, img = o5[0]
                stokes = img.shape[1]
                xdata = img.shape[2]
                ydata = img.shape[3]

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


class OIMS2o5s_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which contains all of the OIMS2o5.py unit
    tests."""
    
    def __init__(self):
        unittest.TestSuite.__init__(self)
        
        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(OIMS2o5_tests)) 


if __name__ == '__main__':
    unittest.main()
