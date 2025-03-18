"""
Unit test for OrvilleImageHDF5 module.
"""

import os
import numpy as np
import tempfile
import unittest

from lsl_toolkits.OrvilleImage import OrvilleImageReader, OrvilleImageHDF5, BAD_FREQ_LIST
from lsl_toolkits.OrvilleImage.legacy import OrvilleImageDB


__version__  = "0.1"
__author__    = "Jayce Dowell"


oimsFile = os.path.join(os.path.dirname(__file__), 'data', 'test.oims')
o5File = os.path.join(os.path.dirname(__file__), 'data', 'test.o5')


class o5_tests(unittest.TestCase):
    """A unittest.TestCase collection of unit tests for the OrvilleImageHDF5
    module."""
    
    testPath = None

    def setUp(self):
        """Turn off all numpy warnings and create the temporary file directory."""

        np.seterr(all='ignore')
        self.testPath = tempfile.mkdtemp(prefix='test-o5-', suffix='.tmp')
        
    def test_o5_read(self):
        """Test reading in an image from a OrvilleImage file."""

        db = OrvilleImageHDF5(o5File, 'r')
        
        # Read in the first image with the correct number of elements
        hdr, data = db.read_image(0)
        ## Image
        self.assertEqual(data.shape[0], db.header.nchan)
        self.assertEqual(data.shape[1], len(db.header.stokes_params.split(',')))
        self.assertEqual(data.shape[2], db.header.ngrid)
        self.assertEqual(data.shape[3], db.header.ngrid)
        
        db.close()
        
    def test_o5_context_manager(self):
        """Test reading in an image from a OrvilleImage file using the context manager."""

        with OrvilleImageHDF5(o5File, 'r') as db:
            # Read in the first image with the correct number of elements
            hdr, data = db[0]
            ## Image
            self.assertEqual(data.shape[0], db.header.nchan)
            self.assertEqual(data.shape[1], len(db.header.stokes_params.split(',')))
            self.assertEqual(data.shape[2], db.header.ngrid)
            self.assertEqual(data.shape[3], db.header.ngrid)
            
    def test_wrapped_o5_read(self):
        with OrvilleImageReader.open(o5File) as db:
            # Read in the first image with the correct number of elements
            hdr, data = db[0]
            ## Image
            self.assertEqual(data.shape[0], db.header.nchan)
            self.assertEqual(data.shape[1], len(db.header.stokes_params.split(',')))
            self.assertEqual(data.shape[2], db.header.ngrid)
            self.assertEqual(data.shape[3], db.header.ngrid)
            
    def test_o5_read_all(self):
        """Test reading in all images from a OrvilleImage file."""

        db = OrvilleImageHDF5(o5File, 'r')
        
        # Read in the first image with the correct number of elements
        hdrs, data = db.read_all()
        ## Count
        self.assertEqual(len(hdrs), 13)
        self.assertEqual(len(hdrs), len(data))
        ## Image
        for d in data:
            self.assertEqual(d.shape[0], db.header.nchan)
            self.assertEqual(d.shape[1], len(db.header.stokes_params.split(',')))
            self.assertEqual(d.shape[2], db.header.ngrid)
            self.assertEqual(d.shape[3], db.header.ngrid)
        
        db.close()
        
    def test_o5_write(self):
        """Test saving data to the OrvilleImageHDF5 format."""
        
        # Setup the file names
        testFile = os.path.join(self.testPath, 'test.o5')
        
        db = OrvilleImageDB(oimsFile, 'r')
        nf = OrvilleImageHDF5(testFile, 'w', imager_version=db.header.imager_version, 
                              station=db.header.station)
                                            
        # Fill it
        for rec in db:
            hdr, img = rec
            for key in hdr:
                if isinstance(hdr[key], bytes):
                    hdr[key] = hdr[key].decode()
            nf.add_image(hdr, img)
            
        # Done
        db.close()
        nf.close()
        
        # Re-open
        db0 = OrvilleImageDB(oimsFile, 'r')
        db1 = OrvilleImageHDF5(testFile, 'r')
        
        # Validate
        ## File header
        for attr in ('imager_version', 'station', 'stokes_params', 'ngrid', 'nchan', 'flags'):
            attr0 = getattr(db0.header, attr, None)
            if isinstance(attr0, bytes):
                attr0 = attr0.decode()
            self.assertEqual(attr0, getattr(db1.header, attr, None))
        for attr in ('pixel_size', 'start_time', 'stop_time'):
            self.assertAlmostEqual(getattr(db0.header, attr, None), getattr(db1.header, attr, None), 6)
        ## First image
        ### Image header
        hdr0, img0 = db0.read_image()
        hdr1, img1 = db1.read_image(0)
        for attr in ('stokes_params', 'ngrid', 'pixel_size', 'ngrid'):
            try:
                attr0 = hdr0[attr]
                if isinstance(attr0, bytes):
                    attr0 = attr0.decode()
            except KeyError:
                attr0 = None
            self.assertEqual(attr0, getattr(hdr1, attr, None))
        for attr in ('start_time', 'int_len', 'lst', 'start_freq', 'stop_freq', 'bandwidth', 'fill', 'center_ra', 'center_dec'):
            try:
                attr0 = hdr0[attr]
                if isinstance(attr0, bytes):
                    attr0 = attr0.decode()
            except KeyError:
                attr0 = None
            self.assertAlmostEqual(attr0, getattr(hdr1, attr, None), 6)
        ### Image
        for i in range(img0.shape[0]):
            for j in range(img0.shape[1]):
                for k in range(img0.shape[2]):
                    for l in range(img0.shape[3]):
                        self.assertAlmostEqual(img0[i,j,k,l], img1[i,j,k,l], 6)                
        ## First image another way
        hdr1, img1 = db1[0]
        for attr in ('stokes_params', 'ngrid', 'pixel_size', 'ngrid'):
            try:
                attr0 = hdr0[attr]
                if isinstance(attr0, bytes):
                    attr0 = attr0.decode()
            except KeyError:
                attr0 = None
            self.assertEqual(attr0, getattr(hdr1, attr, None))
        for attr in ('start_time', 'int_len', 'lst', 'start_freq', 'stop_freq', 'bandwidth', 'fill', 'center_ra', 'center_dec'):
            try:
                attr0 = hdr0[attr]
                if isinstance(attr0, bytes):
                    attr0 = attr0.decode()
            except KeyError:
                attr0 = None
            self.assertAlmostEqual(attr0, getattr(hdr1, attr, None), 6)
        ### Image
        for i in range(img0.shape[0]):
            for j in range(img0.shape[1]):
                for k in range(img0.shape[2]):
                    for l in range(img0.shape[3]):
                        self.assertAlmostEqual(img0[i,j,k,l], img1[i,j,k,l], 6)
                        
        db0.close()
        db1.close()
        
    def test_o5_compressed_write(self):
        """Test saving data to the OrvilleImageHDF5 format with compression."""
        
        # Setup the file names
        testFile = os.path.join(self.testPath, 'test.o5')
        
        db = OrvilleImageDB(oimsFile, 'r')
        nf = OrvilleImageHDF5(testFile, 'w', imager_version=db.header.imager_version, 
                              station=db.header.station, compression='gzip')
                                            
        # Fill it
        for rec in db:
            hdr, img = rec
            for key in hdr:
                if isinstance(hdr[key], bytes):
                    hdr[key] = hdr[key].decode()
            nf.add_image(hdr, img)
            
        # Done
        db.close()
        nf.close()
        
        # Re-open
        db0 = OrvilleImageDB(oimsFile, 'r')
        db1 = OrvilleImageHDF5(testFile, 'r')
        
        # Validate
        ## File header
        for attr in ('imager_version', 'station', 'stokes_params', 'ngrid', 'nchan', 'flags'):
            with self.subTest(file_header_attr=attr):
                attr0 = getattr(db0.header, attr, None)
                if isinstance(attr0, bytes):
                    attr0 = attr0.decode()
                self.assertEqual(attr0, getattr(db1.header, attr, None))
        for attr in ('pixel_size', 'start_time', 'stop_time'):
            with self.subTest(file_header_attr=attr):
                self.assertAlmostEqual(getattr(db0.header, attr, None), getattr(db1.header, attr, None), 6)
        ## First image
        ### Image header
        hdr0, img0 = db0.read_image()
        hdr1, img1 = db1.read_image(0)
        for attr in ('stokes_params', 'ngrid', 'pixel_size', 'ngrid'):
            with self.subTest(frame_header_attr=attr):
                try:
                    attr0 = hdr0[attr]
                    if isinstance(attr0, bytes):
                        attr0 = attr0.decode()
                except KeyError:
                    attr0 = None
                self.assertEqual(attr0, getattr(hdr1, attr, None))
        for attr in ('start_time', 'int_len', 'lst', 'start_freq', 'stop_freq', 'bandwidth', 'fill', 'center_ra', 'center_dec'):
            with self.subTest(frame_header_attr=attr):
                try:
                    attr0 = hdr0[attr]
                    if isinstance(attr0, bytes):
                        attr0 = attr0.decode()
                except KeyError:
                    attr0 = None
                self.assertEqual(attr0, getattr(hdr1, attr, None), 6)
        ### Image
        for i in range(img0.shape[0]):
            for j in range(img0.shape[1]):
                for k in range(img0.shape[2]):
                    for l in range(img0.shape[3]):
                        self.assertAlmostEqual(img0[i,j,k,l], img1[i,j,k,l], 6)                
        ## First image another way
        hdr1, img1 = db1[0]
        for attr in ('stokes_params', 'ngrid', 'pixel_size', 'ngrid'):
            with self.subTest(frame_header_attr=attr):
                try:
                    attr0 = hdr0[attr]
                    if isinstance(attr0, bytes):
                        attr0 = attr0.decode()
                except KeyError:
                    attr0 = None
                self.assertEqual(attr0, getattr(hdr1, attr, None))
        for attr in ('start_time', 'int_len', 'lst', 'start_freq', 'stop_freq', 'bandwidth', 'fill', 'center_ra', 'center_dec'):
            with self.subTest(frame_header_attr=attr):
                try:
                    attr0 = hdr0[attr]
                    if isinstance(attr0, bytes):
                        attr0 = attr0.decode()
                except KeyError:
                    attr0 = None
                self.assertEqual(attr0, getattr(hdr1, attr, None), 6)
        ### Image
        for i in range(img0.shape[0]):
            for j in range(img0.shape[1]):
                for k in range(img0.shape[2]):
                    for l in range(img0.shape[3]):
                        self.assertAlmostEqual(img0[i,j,k,l], img1[i,j,k,l], 6)
                        
        db0.close()
        db1.close()


class o5_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which contains all of the OrvilleImageHDF5
    unit tests."""
    
    def __init__(self):
        unittest.TestSuite.__init__(self)
        
        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(o5_tests)) 


if __name__ == '__main__':
    unittest.main()
