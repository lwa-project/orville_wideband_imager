"""
Unit test for OrvilleImageDB module.
"""

import os
import numpy as np
import tempfile
import unittest

from lsl_toolkits.OrvilleImage import OrvilleImageDB, BAD_FREQ_LIST

try:
    import zfpy
    _HAVE_ZFPY = True
except ImportError:
    _HAVE_ZFPY = False


__version__  = "0.2"
__author__    = "Jayce Dowell"


oimsFileSV = os.path.join(os.path.dirname(__file__), 'data', 'test.oims')
oimsFileNA = os.path.join(os.path.dirname(__file__), 'data', 'test-na.oims')
oimsFile = oimsFileSV


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
            np.testing.assert_array_almost_equal(img0, img1)

            db0.close()
            db1.close()
        
    def test_compression(self):
        """Test that we can read and write OrvilleImageDB files with compression."""
        
        # Setup the file names
        for (cname,cmode,decimal) in [('lossless','fpzip',6),
                                      ('lossy','zfp-tol-0.001',2),
                                      ('lossy', 'zfp-prec-30',4)]:
            if cmode.startswith('zfp-') and not _HAVE_ZFPY:
                continue
                
            with self.subTest(type=cname, method=cmode):
                testFile = os.path.join(self.testPath, 'test.oims')
                try:
                    os.unlink(testFile)
                except OSError:
                    pass
                    
                db = OrvilleImageDB(oimsFile, 'r')
                nf = OrvilleImageDB(testFile, 'w', imager_version=db.header.imager_version, 
                                    station=db.header.station, compression=cmode)
                                                    
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
                np.testing.assert_array_almost_equal(img0, img1, decimal=decimal)
                
                db0.close()
                db1.close()
                
    def test_oims_seek(self):
        """Test seeking within an OrvilleImage file."""
        
        db = OrvilleImageDB(oimsFile, 'r')
        
        # Read all images first
        all_hdrs = []
        all_data = []
        for hdr, data in db:
            all_hdrs.append(hdr)
            all_data.append(data)
            
        # Seek to the last image
        db.seek(db.nint - 1)
        hdr, data = db.read_image()
        self.assertAlmostEqual(hdr['start_time'], all_hdrs[-1]['start_time'], 6)
        np.testing.assert_array_almost_equal(data, all_data[-1])
        
        # Seek to the first image
        db.seek(0)
        hdr, data = db.read_image()
        self.assertAlmostEqual(hdr['start_time'], all_hdrs[0]['start_time'], 6)
        np.testing.assert_array_almost_equal(data, all_data[0])
        
        # Seek to a middle image
        mid = db.nint // 2
        db.seek(mid)
        hdr, data = db.read_image()
        self.assertAlmostEqual(hdr['start_time'], all_hdrs[mid]['start_time'], 6)
        np.testing.assert_array_almost_equal(data, all_data[mid])
        
        # Seek using negative index
        db.seek(-1)
        hdr, data = db.read_image()
        self.assertAlmostEqual(hdr['start_time'], all_hdrs[-1]['start_time'], 6)
        np.testing.assert_array_almost_equal(data, all_data[-1])
        
        db.seek(-db.nint)
        hdr, data = db.read_image()
        self.assertAlmostEqual(hdr['start_time'], all_hdrs[0]['start_time'], 6)
        np.testing.assert_array_almost_equal(data, all_data[0])
        
        # Seek out of range
        self.assertRaises(IndexError, db.seek, db.nint)
        self.assertRaises(IndexError, db.seek, -(db.nint + 1))
        
        db.close()
        
    def test_compression_seek(self):
        """Test seeking within a compressed OrvilleImage file."""
        
        for (cname,cmode,decimal) in [('lossless','fpzip',6),
                                      ('lossy','zfp-tol-0.001',2),
                                      ('lossy', 'zfp-prec-30',4)]:
            if cmode.startswith('zfp-') and not _HAVE_ZFPY:
                continue
                
            with self.subTest(type=cname, method=cmode):
                testFile = os.path.join(self.testPath, 'test.oims')
                try:
                    os.unlink(testFile)
                except OSError:
                    pass
                    
                # Create compressed file
                db = OrvilleImageDB(oimsFile, 'r')
                nf = OrvilleImageDB(testFile, 'w', imager_version=db.header.imager_version,
                                    station=db.header.station, compression=cmode)
                for rec in db:
                    nf.add_image(*rec)
                db.close()
                nf.close()
                
                # Read all from compressed file
                db = OrvilleImageDB(testFile, 'r')
                all_hdrs = []
                all_data = []
                for hdr, data in db:
                    all_hdrs.append(hdr)
                    all_data.append(data)
                    
                # Seek to the last image
                db.seek(db.nint - 1)
                hdr, data = db.read_image()
                self.assertAlmostEqual(hdr['start_time'], all_hdrs[-1]['start_time'], 6)
                np.testing.assert_array_almost_equal(data, all_data[-1], decimal=decimal)
                
                # Seek to the first image
                db.seek(0)
                hdr, data = db.read_image()
                self.assertAlmostEqual(hdr['start_time'], all_hdrs[0]['start_time'], 6)
                np.testing.assert_array_almost_equal(data, all_data[0], decimal=decimal)
                
                # Seek to a middle image
                mid = db.nint // 2
                db.seek(mid)
                hdr, data = db.read_image()
                self.assertAlmostEqual(hdr['start_time'], all_hdrs[mid]['start_time'], 6)
                np.testing.assert_array_almost_equal(data, all_data[mid], decimal=decimal)
                
                # Seek using negative index
                db.seek(-1)
                hdr, data = db.read_image()
                self.assertAlmostEqual(hdr['start_time'], all_hdrs[-1]['start_time'], 6)
                np.testing.assert_array_almost_equal(data, all_data[-1], decimal=decimal)
                
                db.seek(-db.nint)
                hdr, data = db.read_image()
                self.assertAlmostEqual(hdr['start_time'], all_hdrs[0]['start_time'], 6)
                np.testing.assert_array_almost_equal(data, all_data[0], decimal=decimal)
                
                # Seek out of range
                self.assertRaises(IndexError, db.seek, db.nint)
                self.assertRaises(IndexError, db.seek, -(db.nint + 1))
                
                db.close()
        
    def test_sort_compressed(self):
        """Test that sorting works on a compressed file."""
        
        for (cname,cmode,decimal) in [('lossless','fpzip',6),
                                      ('lossy','zfp-tol-0.001',2),
                                      ('lossy', 'zfp-prec-30',4)]:
            if cmode.startswith('zfp-') and not _HAVE_ZFPY:
                continue
                
            with self.subTest(type=cname, method=cmode):
                testFile = os.path.join(self.testPath, 'test.oims')
                try:
                    os.unlink(testFile)
                except OSError:
                    pass
                    
                # Read reference data in original order
                db = OrvilleImageDB(oimsFile, 'r')
                orig_times = []
                for hdr, data in db:
                    orig_times.append(hdr['start_time'])
                    
                # Write a compressed file in reverse order
                nf = OrvilleImageDB(testFile, 'w', imager_version=db.header.imager_version,
                                    station=db.header.station, compression=cmode)
                db.seek(0)
                all_recs = list(db)
                for rec in reversed(all_recs):
                    nf.add_image(*rec)
                db.close()
                nf.close()
                
                # Verify it's not sorted
                db = OrvilleImageDB(testFile, 'r')
                self.assertFalse(db.header.flags & OrvilleImageDB.FLAG_SORTED)
                db.close()
                
                # Sort it
                OrvilleImageDB.sort(testFile)
                
                # Verify the result is sorted and data is intact
                db = OrvilleImageDB(testFile, 'r')
                self.assertTrue(db.header.flags & OrvilleImageDB.FLAG_SORTED)
                self.assertEqual(db.nint, len(orig_times))
                
                prev_time = 0
                for i in range(db.nint):
                    hdr, data = db.read_image()
                    self.assertGreaterEqual(hdr['start_time'], prev_time)
                    prev_time = hdr['start_time']
                    
                db.close()
            
    def test_oims_append_seek(self):
        """Test that seeking works correctly after appending to an existing file."""
        
        testFile = os.path.join(self.testPath, 'test.oims')
        
        # Read the reference data
        db = OrvilleImageDB(oimsFile, 'r')
        all_hdrs = []
        all_data = []
        for hdr, data in db:
            all_hdrs.append(hdr)
            all_data.append(data)
        db.close()
        
        # Write the first half to a new file
        split = len(all_hdrs) // 2
        db = OrvilleImageDB(oimsFile, 'r')
        nf = OrvilleImageDB(testFile, 'w', imager_version=db.header.imager_version,
                            station=db.header.station, compression=None)
        for i in range(split):
            nf.add_image(all_hdrs[i], all_data[i])
        db.close()
        nf.close()
        
        # Append the second half
        nf = OrvilleImageDB(testFile, 'a')
        for i in range(split, len(all_hdrs)):
            nf.add_image(all_hdrs[i], all_data[i])
        nf.close()
        
        # Re-open and verify seeks work across the old/new boundary
        db = OrvilleImageDB(testFile, 'r')
        self.assertEqual(db.nint, len(all_hdrs))
        
        ## Seek to an image from the first half (pre-existing)
        db.seek(0)
        hdr, data = db.read_image()
        self.assertAlmostEqual(hdr['start_time'], all_hdrs[0]['start_time'], 6)
        np.testing.assert_array_almost_equal(data, all_data[0])
        
        ## Seek to an image from the second half (appended)
        db.seek(split)
        hdr, data = db.read_image()
        self.assertAlmostEqual(hdr['start_time'], all_hdrs[split]['start_time'], 6)
        np.testing.assert_array_almost_equal(data, all_data[split])
        
        ## Seek to the last image
        db.seek(-1)
        hdr, data = db.read_image()
        self.assertAlmostEqual(hdr['start_time'], all_hdrs[-1]['start_time'], 6)
        np.testing.assert_array_almost_equal(data, all_data[-1])
        
        db.close()
        
    def test_compression_append_seek(self):
        """Test that seeking works correctly after appending to a compressed file."""
        
        for (cname,cmode,decimal) in [('lossless','fpzip',6),
                                      ('lossy','zfp-tol-0.001',2),
                                      ('lossy', 'zfp-prec-30',4)]:
            if cmode.startswith('zfp-') and not _HAVE_ZFPY:
                continue
                
            with self.subTest(type=cname, method=cmode):
                testFile = os.path.join(self.testPath, 'test.oims')
                try:
                    os.unlink(testFile)
                except OSError:
                    pass
                    
                # Read the reference data
                db = OrvilleImageDB(oimsFile, 'r')
                all_hdrs = []
                all_data = []
                for hdr, data in db:
                    all_hdrs.append(hdr)
                    all_data.append(data)
                db.close()
                
                # Write the first half to a new compressed file
                split = len(all_hdrs) // 2
                db = OrvilleImageDB(oimsFile, 'r')
                nf = OrvilleImageDB(testFile, 'w', imager_version=db.header.imager_version,
                                    station=db.header.station, compression='fpzip')
                for i in range(split):
                    nf.add_image(all_hdrs[i], all_data[i])
                db.close()
                nf.close()
                
                # Append the second half
                nf = OrvilleImageDB(testFile, 'a')
                for i in range(split, len(all_hdrs)):
                    nf.add_image(all_hdrs[i], all_data[i])
                nf.close()
                
                # Re-open and verify seeks work across the old/new boundary
                db = OrvilleImageDB(testFile, 'r')
                self.assertEqual(db.nint, len(all_hdrs))
                
                ## Seek to an image from the first half (pre-existing)
                db.seek(0)
                hdr, data = db.read_image()
                self.assertAlmostEqual(hdr['start_time'], all_hdrs[0]['start_time'], 6)
                np.testing.assert_array_almost_equal(data, all_data[0], decimal=decimal)
                
                ## Seek to an image from the second half (appended)
                db.seek(split)
                hdr, data = db.read_image()
                self.assertAlmostEqual(hdr['start_time'], all_hdrs[split]['start_time'], 6)
                np.testing.assert_array_almost_equal(data, all_data[split], decimal=decimal)
                
                ## Seek to the last image
                db.seek(-1)
                hdr, data = db.read_image()
                self.assertAlmostEqual(hdr['start_time'], all_hdrs[-1]['start_time'], 6)
                np.testing.assert_array_almost_equal(data, all_data[-1], decimal=decimal)
                
                db.close()
            
    def test_compress_static(self):
        """Test the compress() static method for in-place compression."""
        
        for (cname,cmode,decimal) in [('lossless','fpzip',6),
                                      ('lossy','zfp-tol-0.001',2),
                                      ('lossy', 'zfp-prec-30',4)]:
            if cmode.startswith('zfp-') and not _HAVE_ZFPY:
                continue
                
            with self.subTest(type=cname, method=cmode):
                testFile = os.path.join(self.testPath, 'test.oims')
                try:
                    os.unlink(testFile)
                except OSError:
                    pass
                    
                # Write an uncompressed copy
                db = OrvilleImageDB(oimsFile, 'r')
                nf = OrvilleImageDB(testFile, 'w', imager_version=db.header.imager_version,
                                    station=db.header.station, compression=None)
                for rec in db:
                    nf.add_image(*rec)
                db.close()
                nf.close()
                
                # Verify it's not compressed
                db = OrvilleImageDB(testFile, 'r')
                self.assertFalse(db.is_compressed)
                db.close()
                
                # Compress in place
                OrvilleImageDB.compress(testFile, method=cmode)
                
                # Verify it's now compressed and data matches
                db0 = OrvilleImageDB(oimsFile, 'r')
                db1 = OrvilleImageDB(testFile, 'r')
                self.assertTrue(db1.is_compressed)
                self.assertEqual(db0.nint, db1.nint)

                for i in range(db0.nint):
                    hdr0, img0 = db0.read_image()
                    hdr1, img1 = db1.read_image()
                    self.assertAlmostEqual(hdr0['start_time'], hdr1['start_time'], 6)
                    np.testing.assert_array_almost_equal(img0, img1, decimal=decimal)
                    
                db0.close()
                db1.close()
                
    def test_oims_getitem(self):
        """Test the __getitem__ interface for random access."""
        
        db = OrvilleImageDB(oimsFile, 'r')
        
        # Read all sequentially for reference
        all_hdrs = []
        all_data = []
        for hdr, data in db:
            all_hdrs.append(hdr)
            all_data.append(data)
            
        # Access via [] in non-sequential order
        for i in (db.nint - 1, 0, db.nint // 2):
            hdr, data = db[i]
            self.assertAlmostEqual(hdr['start_time'], all_hdrs[i]['start_time'], 6)
            np.testing.assert_array_almost_equal(data, all_data[i])
            
        # Out of range
        self.assertRaises(IndexError, db.__getitem__, db.nint)
        
        db.close()
        
    def test_asp_atten_rename(self):
        """Test that asp_atten_s is returned as asp_atten_3 in info dicts."""

        # Reading the v5 test file should give asp_atten_3, not asp_atten_s
        db = OrvilleImageDB(oimsFile, 'r')
        hdr, data = db.read_image()
        self.assertIn('asp_atten_3', hdr)
        self.assertNotIn('asp_atten_s', hdr)
        db.close()
        
        # Write a new v6 file with asp_atten_3 set explicitly, read it back
        testFile = os.path.join(self.testPath, 'test.oims')
        db = OrvilleImageDB(oimsFile, 'r')
        nf = OrvilleImageDB(testFile, 'w', imager_version=db.header.imager_version,
                            station=db.header.station)
        for hdr, data in db:
            hdr['asp_atten_3'] = 7
            nf.add_image(hdr, data)
        db.close()
        nf.close()
        
        db = OrvilleImageDB(testFile, 'r')
        for hdr, data in db:
            self.assertIn('asp_atten_3', hdr)
            self.assertNotIn('asp_atten_s', hdr)
            self.assertEqual(hdr['asp_atten_3'], 7)
        db.close()
        
    def test_extended_attributes(self):
        """Test writing and reading back extended attributes."""

        testFile = os.path.join(self.testPath, 'test.oims')

        # Read the reference data and attach extended attributes
        db = OrvilleImageDB(oimsFile, 'r')
        all_hdrs = []
        all_data = []
        for hdr, data in db:
            all_hdrs.append(hdr)
            all_data.append(data)
            
        # Write with extended attributes on each integration
        nf = OrvilleImageDB(testFile, 'w', imager_version=db.header.imager_version,
                            station=db.header.station)
        for i, (hdr, data) in enumerate(zip(all_hdrs, all_data)):
            hdr['extended_attributes'] = {'index': i, 'source': 'test', 'values': [1.0, 2.0]}
            nf.add_image(hdr, data)
        db.close()
        nf.close()
        
        # Re-open and verify extended attributes round-trip
        db = OrvilleImageDB(testFile, 'r')
        self.assertEqual(db.nint, len(all_hdrs))
        
        for i in range(db.nint):
            hdr, data = db.read_image()
            self.assertIn('extended_attributes', hdr)
            self.assertEqual(hdr['extended_attributes']['index'], i)
            self.assertEqual(hdr['extended_attributes']['source'], 'test')
            self.assertEqual(hdr['extended_attributes']['values'], [1.0, 2.0])
            
        db.close()
        
    def test_extended_attributes_absent(self):
        """Test that images without extended attributes don't include them."""
        
        testFile = os.path.join(self.testPath, 'test.oims')
        
        db = OrvilleImageDB(oimsFile, 'r')
        nf = OrvilleImageDB(testFile, 'w', imager_version=db.header.imager_version,
                            station=db.header.station)
        for hdr, data in db:
            nf.add_image(hdr, data)
        db.close()
        nf.close()
        
        # Re-open and verify no extended_attributes key
        db = OrvilleImageDB(testFile, 'r')
        for hdr, data in db:
            self.assertNotIn('extended_attributes', hdr)
            self.assertNotIn('extended', hdr)
        db.close()
        
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
