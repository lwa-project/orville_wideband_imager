"""
Unit test for OrvilleImageDB module.
"""

import os
import sys
import glob
import numpy as np
import tempfile
import unittest
import subprocess

from lsl_toolkits.OrvilleImage import OrvilleImageDB
from lsl_toolkits.OrvilleImage.pbtools import pbcorroims
from astropy.io import fits

currentDir = os.path.abspath(os.getcwd())
if os.path.exists(os.path.join(currentDir, 'tests', 'test_OIMS2fits.py')):
    MODULE_BUILD = currentDir
else:
    MODULE_BUILD = None
    
run_scripts_tests = False
if MODULE_BUILD is not None:
    run_scripts_tests = True

__version__  = "0.4"
__author__    = "Jayce Dowell"


oimsFile = os.path.join(os.path.dirname(__file__), 'data', 'test.oims')


@unittest.skipUnless(run_scripts_tests, "cannot determine correct script path to use")
class OIMS2fits_tests(unittest.TestCase):
    """A unittest.TestCase collection of unit tests for the OrvilleImageDB
    module."""
    
    testPath = None

    def setUp(self):
        """Turn off all numpy warnings and create the temporary file directory."""

        np.seterr(all='ignore')
        self.testPath = tempfile.mkdtemp(prefix='test-OIMS2fits-', suffix='.tmp')
        
    def run_OIMS2fits(self, *args, print_on_failure=True):
        """
        Run OIMS2fits.py with the specified arguements and return the subprocess
        status code.
        """
        
        status = 1
        with open('OIMS2fits.log', 'w') as logfile:
            try:
                cmd = [sys.executable, os.path.join(MODULE_BUILD, 'scripts/OIMS2fits.py')]
                cmd.extend(['-o', self.testPath])
                cmd.extend(args)
                
                status = subprocess.check_call(cmd, stdout=logfile)
            except subprocess.CalledProcessError:
                pass
                
        if status == 1 and print_on_failure:
            with open('OIMS2fits.log', 'r') as logfile:
                print(logfile.read())
                
        os.unlink('OIMS2fits.log')
        
        return status
        
    def test_OIMS2fitsdef_run(self):
        """Create fits from oims with default settings"""
        
        status = self.run_OIMS2fits(oimsFile)
        self.assertEqual(status, 0)
        
        with OrvilleImageDB(oimsFile, 'r') as db:
            fitsFile = np.sort(glob.glob(os.path.join(self.testPath, '*.fits')))
            db.seek(0)
            hdr, data = db.read_image()
            oimsfreqs = np.array([(hdr['start_freq'] + (c*hdr['bandwidth'])) for c in range(db.header.nchan)])
            fitsfreqs = np.zeros(len(oimsfreqs))
            nchan = len(fitsFile)
            for c,f in enumerate(fitsFile):
                with fits.open(f) as hdul:
                    for i,hdu in enumerate(hdul):
                        ints = len(hdul)
                        stokes = hdu.data.shape[0]
                        xdata = hdu.data.shape[1]
                        ydata = hdu.data.shape[2]
                        curfitsfreq = hdu.header['RESTFREQ']
                        fitsfreqs[c] = curfitsfreq
                        db.seek(i)
                        hdr, data = db.read_image()
                        data = np.copy(np.asarray(data))
                        ## Zero outside of the horizon so avoid problems
                        sRad   = 360.0/db.header.pixel_size/np.pi / 2
                        x = np.arange(data.shape[-2]) - 0.5
                        y = np.arange(data.shape[-1]) - 0.5
                        x,y = np.meshgrid(x,y)
                        invalid = np.where( ((x-db.header.ngrid/2.0)**2 + (y-db.header.ngrid/2.0)**2) > (sRad**2) )
                        data[:,:,invalid[0], invalid[1]] = 0.0
                        curoimsdata = np.squeeze(data[curfitsfreq==oimsfreqs,:,:,:])
                        np.testing.assert_array_equal(hdu.data,curoimsdata)
                        self.assertEqual(ints, db.nint)
                        self.assertEqual(stokes, len(db.header.stokes_params.split(b',')))
                        self.assertEqual(xdata, db.header.ngrid)
                        self.assertEqual(ydata, db.header.ngrid)
            self.assertEqual(nchan, db.header.nchan)
            np.testing.assert_array_equal(np.sort(oimsfreqs),np.sort(fitsfreqs)) 
        
    def test_OIMS2fitspbcor_run(self):
        """Create fits from oims with pbcorr"""
        
        status = self.run_OIMS2fits('-p', oimsFile)
        self.assertEqual(status, 0)
        
        with OrvilleImageDB(oimsFile, 'r') as db:
            fitsFile = np.sort(glob.glob(os.path.join(self.testPath, '*.fits')))
            db.seek(0)
            hdr, data = db.read_image()
            oimsfreqs = np.array([(hdr['start_freq'] + (c*hdr['bandwidth'])) for c in range(db.header.nchan)])
            fitsfreqs = np.zeros(len(oimsfreqs))
            nchan = len(fitsFile)
            for c,f in enumerate(fitsFile):
                with fits.open(f) as hdul:
                    for i,hdu in enumerate(hdul):
                        ints = len(hdul)
                        stokes = hdu.data.shape[0]
                        xdata = hdu.data.shape[1]
                        ydata = hdu.data.shape[2]
                        curfitsfreq = hdu.header['RESTFREQ']
                        fitsfreqs[c] = curfitsfreq
                        db.seek(i)
                        hdr, data = db.read_image()
                        data = np.copy(np.asarray(data))
                        ## Zero outside of the horizon so avoid problems
                        sRad   = 360.0/db.header.pixel_size/np.pi / 2
                        x = np.arange(data.shape[-2]) - 0.5
                        y = np.arange(data.shape[-1]) - 0.5
                        x,y = np.meshgrid(x,y)
                        invalid = np.where( ((x-db.header.ngrid/2.0)**2 + (y-db.header.ngrid/2.0)**2) > (sRad**2) )
                        data[:,:,invalid[0], invalid[1]] = 0.0
                        curoimsdata = np.squeeze(data[curfitsfreq==oimsfreqs,:,:,:])
                        XX,YY = pbcorroims(hdr,db.header.ngrid,c,db.header.station)
                        curoimsdata[0]/=((XX+YY)/2)
                        np.testing.assert_array_equal(hdu.data,curoimsdata)
                        self.assertEqual(ints, db.nint)
                        self.assertEqual(stokes, len(db.header.stokes_params.split(b',')))
                        self.assertEqual(xdata, db.header.ngrid)
                        self.assertEqual(ydata, db.header.ngrid)
            self.assertEqual(nchan, db.header.nchan)
            np.testing.assert_array_equal(np.sort(oimsfreqs),np.sort(fitsfreqs)) 

    def test_OIMS2fitsindex_run(self):
        """Create fits from oims with specified index"""
        
        status = self.run_OIMS2fits('-i 2', oimsFile)
        self.assertEqual(status, 0)
        
        fitsFile = np.sort(glob.glob(os.path.join(self.testPath, '*.fits')))
        knownpix = np.array([62.473182678222656,75.6445083618164,77.76968383789062,74.6494369506836,64.94345092773438,54.68968200683594])
        testpix = np.zeros(knownpix.shape)
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
                
        np.testing.assert_array_equal(testpix,knownpix)
        
    def test_OIMS2fitsdiff_run(self):
        """Create fits from oims with diff ims"""
        
        status = self.run_OIMS2fits('-d', oimsFile)
        self.assertEqual(status, 0)
        
        fitsFile = np.sort(glob.glob(os.path.join(self.testPath, '*.fits')))
        knownpix = np.array([-3.5097122192382812,-2.56549072265625,-1.3171768188476562,-3.9064254760742188,-6.341712951660156,-10.29165267944336])

        testpix = np.zeros(knownpix.shape)
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
                
        np.testing.assert_allclose(testpix,knownpix)

    def test_OIMS2fitschan_run(self):
        """Create fits from oims with specified channel"""
        
        status = self.run_OIMS2fits('--channel', '0', oimsFile)
        self.assertEqual(status, 0)
        
        fitsFile = np.sort(glob.glob(os.path.join(self.testPath, '*.fits')))
        knownpix = np.array([69.75801086425781])
        testpix = np.zeros(knownpix.shape)
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
        np.testing.assert_array_equal(testpix,knownpix)
        
    def test_OIMS2fitscorrfacback_run(self):
        """Create fits from oims with specified corrfac and background"""
        
        status = self.run_OIMS2fits('-b 5', '-c 100', oimsFile)
        self.assertEqual(status, 0)
        
        fitsFile = np.sort(glob.glob(os.path.join(self.testPath, '*.fits')))
        knownpix = np.array([6475.801086425781,7563.559722900391,7490.340423583984,7126.630401611328,6334.429931640625,5794.603729248047])
        testpix = np.zeros(knownpix.shape)
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
        np.testing.assert_allclose(testpix,knownpix)
        
    def tearDown(self):
        """Remove the test path directory and its contents"""
        
        tempFiles = os.listdir(self.testPath)
        for tempFile in tempFiles:
            os.unlink(os.path.join(self.testPath, tempFile))
        os.rmdir(self.testPath)
        self.testPath = None


class OIMS2fits_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which contains all of the OrvilleImageDB units 
    tests."""
    
    def __init__(self):
        unittest.TestSuite.__init__(self)
        
        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(OIMS2fits_tests)) 


if __name__ == '__main__':
    unittest.main()
