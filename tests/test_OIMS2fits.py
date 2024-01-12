"""
Unit test for OrvilleImageDB module.
"""

# Python2 compatibility
from __future__ import print_function, division, absolute_import
try:
    range = xrange
except NameError:
    pass
    
import os
import numpy
import tempfile
import unittest
from argparse import Namespace
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import OrvilleImageDB
from scripts import OIMS2fits
from astropy.io import fits

__version__  = "0.2"
__author__    = "Jayce Dowell"


oimsFile = os.path.join(os.path.dirname(__file__), 'data', 'test.oims')

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
        args = Namespace(filename=[oimsFile],diff=False,force=False,pbcorr=False,verbose=False)
        fitsFile = oimsFile.replace(".oims",".fits")
        try:
            os.remove(fitsFile)
        except OSError:
            pass
        OIMS2fits.main(args)
        with fits.open(fitsFile) as hdul:
            nchan = len(hdul)
            ints = hdul[0].data.shape[0]
            stokes = hdul[0].data.shape[1]
            xdata = hdul[0].data.shape[2]
            ydata = hdul[0].data.shape[3]

        db = OrvilleImageDB.OrvilleImageDB(oimsFile, 'r')
        self.assertEqual(nchan, db.header.nchan)
        self.assertEqual(ints, db.nint)
        self.assertEqual(stokes, len(db.header.stokes_params.split(b',')))
        self.assertEqual(xdata, db.header.ngrid)
        self.assertEqual(ydata, db.header.ngrid)
        db.close()
        try:
            os.remove(fitsFile)
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
