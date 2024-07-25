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

from lsl_toolkits.OrvilleImage import OrvilleImageDB
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
    def test_OIMS2fitsdef_run(self):
        """Create fits from oims with default settings"""
        
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
        knownpix = numpy.array([-33.82817840576172,-28.67790412902832,-27.860015869140625,-37.22902297973633,-21.941648483276367,-19.186431884765625])
        testpix = numpy.zeros(knownpix.shape)
        for i,f in enumerate(fitsFile):
            with fits.open(f) as hdul:
                nchan = len(fitsFile)
                ints = len(hdul)
                stokes = hdul[0].data.shape[0]
                xdata = hdul[0].data.shape[1]
                ydata = hdul[0].data.shape[2]
                testpix[i] = hdul[0].data[0,ydata//2,xdata//2]
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
        numpy.testing.assert_array_equal(testpix,knownpix)
        
    def test_OIMS2fitspbcor_run(self):
        """Create fits from oims with pbcorr"""
        
        fitsFile = glob.glob(oimsFile.replace(".oims", "*.fits"))
        if fitsFile:
            for f in fitsFile:
                try:
                    os.remove(f)
                except OSError:
                    pass
                    
        with open('OIMS2fits.log', 'w') as logfile:
            try:
                cmd = [sys.executable, 'scripts/OIMS2fits.py','-p', oimsFile]
                status = subprocess.check_call(cmd, stdout=logfile, stderr=logfile)
            except subprocess.CalledProcessError:
                status = 1
        if status == 1:
            with open('OIMS2fits.log', 'r') as logfile:
                print(logfile.read())
        os.unlink('OIMS2fits.log')
        self.assertEqual(status, 0)
        
        fitsFile = glob.glob(oimsFile.replace(".oims", "*.fits"))
        knownpix = numpy.array([188.95354803841997,169.35453431526005,200.2454557769261,198.07150760209828,156.03151446855847,173.84442224017718])
        testpix = numpy.zeros(knownpix.shape)
        for i,f in enumerate(fitsFile):
            with fits.open(f) as hdul:
                nchan = len(fitsFile)
                ints = len(hdul)
                stokes = hdul[0].data.shape[0]
                xdata = hdul[0].data.shape[1]
                ydata = hdul[0].data.shape[2]
                testpix[i] = hdul[0].data[0,ydata//4,xdata//4]

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
        numpy.testing.assert_array_equal(testpix,knownpix)

    def test_OIMS2fitsindex_run(self):
        """Create fits from oims with specified index"""
        
        fitsFile = glob.glob(oimsFile.replace(".oims", "*.fits"))
        if fitsFile:
            for f in fitsFile:
                try:
                    os.remove(f)
                except OSError:
                    pass
                    
        with open('OIMS2fits.log', 'w') as logfile:
            try:
                cmd = [sys.executable, 'scripts/OIMS2fits.py','-i 2', oimsFile]
                status = subprocess.check_call(cmd, stdout=logfile)
            except subprocess.CalledProcessError:
                status = 1
        if status == 1:
            with open('OIMS2fits.log', 'r') as logfile:
                print(logfile.read())
        os.unlink('OIMS2fits.log')
        self.assertEqual(status, 0)
        
        fitsFile = glob.glob(oimsFile.replace(".oims", "*.fits"))
        knownpix = numpy.array([54.68968200683594,64.94345092773438,62.473182678222656,77.76968383789062,74.6494369506836,75.6445083618164])
        testpix = numpy.zeros(knownpix.shape)
        for i,f in enumerate(fitsFile):
            with fits.open(f) as hdul:
                nchan = len(fitsFile)
                ints = len(hdul)
                stokes = hdul[0].data.shape[0]
                xdata = hdul[0].data.shape[1]
                ydata = hdul[0].data.shape[2]
                testpix[i] = hdul[0].data[0,ydata//4,xdata//4]

            with OrvilleImageDB(oimsFile, 'r') as db:
                self.assertEqual(nchan, db.header.nchan)
                self.assertEqual(ints, 1)
                self.assertEqual(stokes, len(db.header.stokes_params.split(b',')))
                self.assertEqual(xdata, db.header.ngrid)
                self.assertEqual(ydata, db.header.ngrid)
                
            try:
                os.remove(f)
            except OSError:
                pass
        numpy.testing.assert_array_equal(testpix,knownpix)
    def test_OIMS2fitsdiff_run(self):
        """Create fits from oims with diff ims"""
        
        fitsFile = glob.glob(oimsFile.replace(".oims", "*.fits"))
        if fitsFile:
            for f in fitsFile:
                try:
                    os.remove(f)
                except OSError:
                    pass
                    
        with open('OIMS2fits.log', 'w') as logfile:
            try:
                cmd = [sys.executable, 'scripts/OIMS2fits.py','-d', oimsFile]
                status = subprocess.check_call(cmd, stdout=logfile)
            except subprocess.CalledProcessError:
                status = 1
        if status == 1:
            with open('OIMS2fits.log', 'r') as logfile:
                print(logfile.read())
        os.unlink('OIMS2fits.log')
        self.assertEqual(status, 0)
        
        fitsFile = glob.glob(oimsFile.replace(".oims", "*.fits"))
        knownpix = numpy.array([-3.9064254760742188,-10.29165267944336,-6.341712951660156,-1.3171768188476562,-3.5097122192382812,-2.56549072265625])
        testpix = numpy.zeros(knownpix.shape)
        for i,f in enumerate(fitsFile):
            with fits.open(f) as hdul:
                nchan = len(fitsFile)
                ints = len(hdul)
                stokes = hdul[0].data.shape[0]
                xdata = hdul[0].data.shape[1]
                ydata = hdul[0].data.shape[2]
                testpix[i] = hdul[0].data[0,ydata//4,xdata//4]

            with OrvilleImageDB(oimsFile, 'r') as db:
                self.assertEqual(nchan, db.header.nchan)
                self.assertEqual(ints, db.nint-1)
                self.assertEqual(stokes, len(db.header.stokes_params.split(b',')))
                self.assertEqual(xdata, db.header.ngrid)
                self.assertEqual(ydata, db.header.ngrid)
                
            try:
                os.remove(f)
            except OSError:
                pass
        numpy.testing.assert_array_equal(testpix,knownpix)

    def test_OIMS2fitschan_run(self):
        """Create fits from oims with specified channel"""
        
        fitsFile = glob.glob(oimsFile.replace(".oims", "*.fits"))
        if fitsFile:
            for f in fitsFile:
                try:
                    os.remove(f)
                except OSError:
                    pass
                    
        with open('OIMS2fits.log', 'w') as logfile:
            try:
                cmd = [sys.executable, 'scripts/OIMS2fits.py','--channel','0', oimsFile]
                status = subprocess.check_call(cmd, stdout=logfile)
            except subprocess.CalledProcessError:
                status = 1
        if status == 1:
            with open('OIMS2fits.log', 'r') as logfile:
                print(logfile.read())
        os.unlink('OIMS2fits.log')
        self.assertEqual(status, 0)
        
        fitsFile = glob.glob(oimsFile.replace(".oims", "*.fits"))
        knownpix = numpy.array([69.75801086425781])
        testpix = numpy.zeros(knownpix.shape)
        for i,f in enumerate(fitsFile):
            with fits.open(f) as hdul:
                nchan = len(fitsFile)
                ints = len(hdul)
                stokes = hdul[0].data.shape[0]
                xdata = hdul[0].data.shape[1]
                ydata = hdul[0].data.shape[2]
                testpix[i] = hdul[0].data[0,ydata//4,xdata//4]

            with OrvilleImageDB(oimsFile, 'r') as db:
                self.assertEqual(nchan, 1)
                self.assertEqual(ints, db.nint)
                self.assertEqual(stokes, len(db.header.stokes_params.split(b',')))
                self.assertEqual(xdata, db.header.ngrid)
                self.assertEqual(ydata, db.header.ngrid)
                
            try:
                os.remove(f)
            except OSError:
                pass
        numpy.testing.assert_array_equal(testpix,knownpix)
    def test_OIMS2fitscorrfacback_run(self):
        """Create fits from oims with specified corrfac and background"""
        
        fitsFile = glob.glob(oimsFile.replace(".oims", "*.fits"))
        if fitsFile:
            for f in fitsFile:
                try:
                    os.remove(f)
                except OSError:
                    pass
                    
        with open('OIMS2fits.log', 'w') as logfile:
            try:
                cmd = [sys.executable, 'scripts/OIMS2fits.py','-b 5','-c 100', oimsFile]
                status = subprocess.check_call(cmd, stdout=logfile)
            except subprocess.CalledProcessError:
                status = 1
        if status == 1:
            with open('OIMS2fits.log', 'r') as logfile:
                print(logfile.read())
        os.unlink('OIMS2fits.log')
        self.assertEqual(status, 0)
        
        fitsFile = glob.glob(oimsFile.replace(".oims", "*.fits"))
        knownpix = numpy.array([7126.630401611328,6334.429931640625,7563.559722900391,7490.340423583984,5794.603729248047,6475.801086425781])
        testpix = numpy.zeros(knownpix.shape)
        for i,f in enumerate(fitsFile):
            with fits.open(f) as hdul:
                nchan = len(fitsFile)
                ints = len(hdul)
                stokes = hdul[0].data.shape[0]
                xdata = hdul[0].data.shape[1]
                ydata = hdul[0].data.shape[2]
                testpix[i] = hdul[0].data[0,ydata//4,xdata//4]

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
        numpy.testing.assert_array_equal(testpix,knownpix)

class OIMS2fits_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which contains all of the OrvilleImageDB units 
    tests."""
    
    def __init__(self):
        unittest.TestSuite.__init__(self)
        
        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(OIMS2fits_tests)) 


if __name__ == '__main__':
    unittest.main()
