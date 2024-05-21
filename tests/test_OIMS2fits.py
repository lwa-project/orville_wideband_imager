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
import glob
import numpy
import tempfile
import unittest
from argparse import Namespace
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from OrvilleImager import OrvilleImageDB
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
        args = Namespace(filename=[oimsFile],background=0,corrfac=1,channel=None,diff=False,force=False,index=None,pbcorr=False,verbose=False)
        fitsFile = glob.glob(oimsFile.replace(".oims","*.fits"))
        if fitsFile:
            for f in fitsFile:
                try:
                    os.remove(f)
                except OSError:
                    pass
        OIMS2fits.main(args)
        fitsFile = glob.glob(oimsFile.replace(".oims","*.fits"))
        for f in fitsFile:
            with fits.open(f) as hdul:
                nchan = len(fitsFile)
                ints = len(hdul)
                stokes = hdul[0].data.shape[0]
                xdata = hdul[0].data.shape[1]
                ydata = hdul[0].data.shape[2]

            db = OrvilleImageDB.OrvilleImageDB(oimsFile, 'r')
            self.assertEqual(nchan, db.header.nchan)
            self.assertEqual(ints, db.nint)
            self.assertEqual(stokes, len(db.header.stokes_params.split(b',')))
            self.assertEqual(xdata, db.header.ngrid)
            self.assertEqual(ydata, db.header.ngrid)
            db.close()
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
