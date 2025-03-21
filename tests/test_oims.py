"""
Unit test for OrvilleImageDB module.
"""

import os
import numpy as np
import tempfile
import unittest

from lsl_toolkits.OrvilleImage import OrvilleImageDB, BAD_FREQ_LIST


__version__  = "0.2"
__author__    = "Jayce Dowell"


oimsFileSV = os.path.join(os.path.dirname(__file__), 'data', 'test.oims')
oimsFileNA = os.path.join(os.path.dirname(__file__), 'data', 'test-na.oims')


class oims_tests(unittest.TestCase):
    """A unittest.TestCase collection of unit tests for the OrvilleImageDB
    module."""
    
    testPath = None

    def setUp(self):
        """Turn off all numpy warnings and create the temporary file directory."""

        np.seterr(all='ignore')
        self.testPath = tempfile.mkdtemp(prefix='test-oims-', suffix='.tmp')
        
    def test_oims_read(self):
        """Test reading in an image from a OrvilleImage file."""

        for oimsFile in (oimsFileSV, oimsFileNA):
            db = OrvilleImageDB(oimsFile, 'r')
            
            # Read in the first image with the correct number of elements
            hdr, data = db.read_image()
            ## Image
            self.assertEqual(data.shape[0], db.header.nchan)
            self.assertEqual(data.shape[1], len(db.header.stokes_params.split(b',')))
            self.assertEqual(data.shape[2], db.header.ngrid)
            self.assertEqual(data.shape[3], db.header.ngrid)
            
            db.close()
            
    def test_oims_context_manager(self):
        """Test reading in an image from a OrvilleImage file using the context manager."""
        
        for oimsFile in (oimsFileSV, oimsFileNA):
            with OrvilleImageDB(oimsFile, 'r') as db:
                # Read in the first image with the correct number of elements
                hdr, data = db.read_image()
                ## Image
                self.assertEqual(data.shape[0], db.header.nchan)
                self.assertEqual(data.shape[1], len(db.header.stokes_params.split(b',')))
                self.assertEqual(data.shape[2], db.header.ngrid)
                self.assertEqual(data.shape[3], db.header.ngrid)
                
    def test_oims_read_all(self):
        """Test reading in all images from a OrvilleImage file."""

        for oimsFile in (oimsFileSV, oimsFileNA):
            db = OrvilleImageDB(oimsFile, 'r')
            
            # Read in the first image with the correct number of elements
            hdrs, data = db.read_all()
            ## Count
            self.assertEqual(len(hdrs), 13)
            self.assertEqual(len(hdrs), len(data))
            ## Image
            for d in data:
                self.assertEqual(d.shape[0], db.header.nchan)
                self.assertEqual(d.shape[1], len(db.header.stokes_params.split(b',')))
                self.assertEqual(d.shape[2], db.header.ngrid)
                self.assertEqual(d.shape[3], db.header.ngrid)
            
            db.close()
            
    def test_oims_loop(self):
        """Test reading in a collection of images in a loop."""
        
        for oimsFile in (oimsFileSV, oimsFileNA):
            db = OrvilleImageDB(oimsFile, 'r')
            
            # Go
            for i,(hdr,data) in enumerate(db):
                i
                
            db.close()
        
    def test_oims_write(self):
        """Test saving data to the OrvilleImageDB format."""
        
        for oimsFile in (oimsFileSV, oimsFileNA):
            # Setup the file names
            testFile = os.path.join(self.testPath, os.path.basename(oimsFile))
            
            db = OrvilleImageDB(oimsFile, 'r')
            nf = OrvilleImageDB(testFile, 'w', imager_version=db.header.imager_version, 
                                station=db.header.station)
                                                
            # Fill it
            for rec in db:
                nf.add_image(*rec)
                
            # Done
            db.close()
            nf.close()
            
            # Re-open
            db0 = OrvilleImageDB(oimsFile, 'r')
            db1 = OrvilleImageDB(testFile, 'r')
            
            # Validate
            ## File header
            for attr in ('imager_version', 'station', 'stokes_params', 'ngrid', 'nchan', 'flags'):
                self.assertEqual(getattr(db0.header, attr, None), getattr(db1.header, attr, None))
            for attr in ('pixel_size', 'start_time', 'stop_time'):
                self.assertAlmostEqual(getattr(db0.header, attr, None), getattr(db1.header, attr, None), 6)
            ## First image
            ### Image header
            hdr0, img0 = db0.read_image()
            hdr1, img1 = db1.read_image()
            for attr in ('stokes_params', 'ngrid', 'pixel_size', 'ngrid'):
                self.assertEqual(getattr(hdr0, attr, None), getattr(hdr1, attr, None))
            for attr in ('start_time', 'int_len', 'lst', 'start_freq', 'stop_freq', 'bandwidth', 'fill', 'center_ra', 'center_dec'):
                self.assertAlmostEqual(getattr(hdr0, attr, None), getattr(hdr1, attr, None), 6)
            ### Image
            for i in range(img0.shape[0]):
                for j in range(img0.shape[1]):
                    for k in range(img0.shape[2]):
                        for l in range(img0.shape[3]):
                            self.assertAlmostEqual(img0[i,j,k,l], img1[i,j,k,l], 6)
                            
            db0.close()
            db1.close()
            
    def test_bad_freq_list(self):
        """Test the list of bad frequencies that should be flagged/removed."""
        
        self.assertTrue(BAD_FREQ_LIST.size > 0)
        
    def tearDown(self):
        """Remove the test path directory and its contents"""
        
        tempFiles = os.listdir(self.testPath)
        for tempFile in tempFiles:
            os.unlink(os.path.join(self.testPath, tempFile))
        os.rmdir(self.testPath)
        self.testPath = None


class oims_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which contains all of the OrvilleImageDB units 
    tests."""
    
    def __init__(self):
        unittest.TestSuite.__init__(self)
        
        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(oims_tests)) 


if __name__ == '__main__':
    unittest.main()
