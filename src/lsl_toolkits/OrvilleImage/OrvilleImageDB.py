import os
import sys
import gzip
import json
import numpy as np
import ctypes
import struct
import shutil
import tempfile

import fpzip
import zfpy

from lsl.common.progress import ProgressBarPlus


def _get_compression_method(name):
    """
    Convert a compression method name into a compressor and decompressor
    function.  If the compression name is none then None is returned for
    both functions.  Any unidentified methods raise a ValueError.
    """
    
    try:
        name = name.decode()
    except (TypeError, AttributeError):
        pass
        
    comp, decomp = None, None
    if name in ('None', 'none') or name is None:
        # No compression requested
        pass
        
    elif name == 'fpzip':
        # Lossless compression with fpzip
        comp = lambda data: \
                   fpzip.compress(data.astype('<f4').transpose(1,0,2,3), \
                                  order='C')
        decomp = lambda cdata: \
                     fpzip.decompress(cdata).transpose(1,0,2,3)
        
    elif name.startswith('zfp-'):
        # Lossy compression with zfpy
        _, param, value = name.split('-', 2)
        try:
            value = int(value, 10)
        except ValueError:
            value = float(value)
            
        if param == 'tol':
            ## Fixed accurary mode
            comp = lambda data: \
                       zfpy.compress_numpy(data.astype('<f4').transpose(1,0,2,3), \
                                           tolerance=value, write_header=True)
        elif param == 'prec':
            ## Fixed precision mode
            comp = lambda data: \
                       zfpy.compress_numpy(data.astype('<f4').transpose(1,0,2,3), \
                                           precision=value, write_header=True)
        else:
            raise ValueError("Unsupported ZFP mode '%s'" % name)
        decomp = lambda cdata: \
                     zfpy.decompress_numpy(cdata).transpose(1,0,2,3)
        
    else:
        raise ValueError("Unsupported compression mode '%s'" % name)
        
    return comp, decomp


class PrintableLittleEndianStructure(ctypes.LittleEndianStructure):
    """
    Sub-class of ctypes.LittleEndianStructure that adds a as_dict()
    method for accessing all of the fields as a dictionary.
    """
    
    def as_dict(self):
        """
        Return all of the structure fields as a dictionary.
        """
        
        out = {}
        for field in self._fields_:
            out[field[0]] = getattr(self, field[0], None)
        return out
        
    def __repr__(self):
        return repr(self.as_dict())


class OrvilleImageDB(object):
    """
    Encapsulates a OrvilleImageDB binary file.
    
    This class can be used for both reading and writing OrvilleImageDB files.
    For reading, initialize with mode = "r" and use the read_image() function.  
    For writing, use mode = "a" or "w" and the add_image() function.  Be sure 
    to always call the close() method after adding images in order to update 
    the file's header information.
    
    Public module variables:
      header -- a class with member variables describing the data, including:
        imager_version, imager version used to create the images
        station, the station name
        stokes_params, the comma-delimited parameters (e.g., 'I,Q,U,V')
        ngrid, the dimensions of the image in pixels
        pixel_size, the physical size of a pixel, in degrees
        nchan, the number of channels for each image
        flags, a bitfield: 0x1 = sorted; others zero
        start_time, the earliest time of data covered by this file, in MJD UTC
        stop_time, the latest time of data, in MJD UTC
    """
    
    # The OrvilleImageDB files start with a 24 byte string that specifies the
    # data file's format version.  Following it is that format's header block,
    # defined by the _FileHeader structure.  After the header are the 
    # integration blocks.  An integration block starts with a header, defined
    # by the _EntryHeader structure.  Following that are the images, packed as 
    # [chan, stokes, x, y] and float32s.  The alignment of the images is little 
    # endian.
    #
    # All absolute times are in MJD UTC, LSTs are in days (i.e., from 0 to
    # 0.99726957), and integration lengths are in days.  Sky directions
    # (including RA) and pixel sizes are in degrees.  All other entries are in
    # standard mks units.
    
    _FORMAT_VERSION = 'OrvilleImageDBv006'
    
    class _FileHeader_v1(PrintableLittleEndianStructure):
        _pack_   = 1
        _fields_ = [('imager_version', ctypes.c_char*24),
                    ('station',        ctypes.c_char*24),
                    ('stokes_params',  ctypes.c_char*24),
                    ('ngrid',          ctypes.c_int),
                    ('pixel_size',     ctypes.c_double),
                    ('nchan',          ctypes.c_int),
                    ('flags',          ctypes.c_uint),
                    ('start_time',     ctypes.c_double),
                    ('stop_time',      ctypes.c_double)]
    _FileHeader_v2 = _FileHeader_v1
    _FileHeader_v3 = _FileHeader_v2
    _FileHeader_v4 = _FileHeader_v3
    _FileHeader_v5 = _FileHeader_v4
    class _FileHeader_v6(PrintableLittleEndianStructure):
        _pack_   = 1
        _fields_ = [('imager_version', ctypes.c_char*24),
                    ('station',        ctypes.c_char*24),
                    ('stokes_params',  ctypes.c_char*24),
                    ('ngrid',          ctypes.c_int),
                    ('pixel_size',     ctypes.c_double),
                    ('nchan',          ctypes.c_int),
                    ('flags',          ctypes.c_uint),
                    ('start_time',     ctypes.c_double),
                    ('stop_time',      ctypes.c_double),
                    ('compression',    ctypes.c_char*24)]
    
    FLAG_SORTED = 0x0001

    class _EntryHeader_v1(PrintableLittleEndianStructure):
        _pack_   = 1
        _fields_ = [('sync_word',  ctypes.c_uint),
                    ('start_time', ctypes.c_double),
                    ('int_len',    ctypes.c_double),
                    ('lst',        ctypes.c_double),
                    ('start_freq', ctypes.c_double),
                    ('stop_freq',  ctypes.c_double),
                    ('bandwidth',  ctypes.c_double),
                    ('center_ra',  ctypes.c_double),
                    ('center_dec', ctypes.c_double)]
    class _EntryHeader_v2(PrintableLittleEndianStructure):
        _pack_   = 1
        _fields_ = [('sync_word',  ctypes.c_uint),
                    ('start_time', ctypes.c_double),
                    ('int_len',    ctypes.c_double),
                    ('fill',       ctypes.c_double),
                    ('lst',        ctypes.c_double),
                    ('start_freq', ctypes.c_double),
                    ('stop_freq',  ctypes.c_double),
                    ('bandwidth',  ctypes.c_double),
                    ('center_ra',  ctypes.c_double),
                    ('center_dec', ctypes.c_double),
                    ('center_az',  ctypes.c_double),
                    ('center_alt', ctypes.c_double)]
    _EntryHeader_v3 = _EntryHeader_v2
    class _EntryHeader_v4(PrintableLittleEndianStructure):
        _pack_   = 1
        _fields_ = [('sync_word',   ctypes.c_uint),
                    ('start_time',  ctypes.c_double),
                    ('int_len',     ctypes.c_double),
                    ('fill',        ctypes.c_double),
                    ('lst',         ctypes.c_double),
                    ('start_freq',  ctypes.c_double),
                    ('stop_freq',   ctypes.c_double),
                    ('bandwidth',   ctypes.c_double),
                    ('center_ra',   ctypes.c_double),
                    ('center_dec',  ctypes.c_double),
                    ('center_az',   ctypes.c_double),
                    ('center_alt',  ctypes.c_double),
                    ('asp_filter',  ctypes.c_int),
                    ('asp_atten_1', ctypes.c_int),
                    ('asp_atten_2', ctypes.c_int),
                    ('asp_atten_s', ctypes.c_int)]
    class _EntryHeader_v5(PrintableLittleEndianStructure):
        _pack_   = 1
        _fields_ = [('sync_word',   ctypes.c_uint),
                    ('start_time',  ctypes.c_double),
                    ('int_len',     ctypes.c_double),
                    ('fill',        ctypes.c_double),
                    ('lst',         ctypes.c_double),
                    ('start_freq',  ctypes.c_double),
                    ('stop_freq',   ctypes.c_double),
                    ('bandwidth',   ctypes.c_double),
                    ('weighting',   ctypes.c_char*24),
                    ('center_ra',   ctypes.c_double),
                    ('center_dec',  ctypes.c_double),
                    ('center_az',   ctypes.c_double),
                    ('center_alt',  ctypes.c_double),
                    ('asp_filter',  ctypes.c_int),
                    ('asp_atten_1', ctypes.c_int),
                    ('asp_atten_2', ctypes.c_int),
                    ('asp_atten_s', ctypes.c_int)]
    class _EntryHeader_v6(PrintableLittleEndianStructure):
        _pack_   = 1
        _fields_ = [('sync_word',   ctypes.c_uint),
                    ('start_time',  ctypes.c_double),
                    ('int_len',     ctypes.c_double),
                    ('fill',        ctypes.c_double),
                    ('lst',         ctypes.c_double),
                    ('start_freq',  ctypes.c_double),
                    ('stop_freq',   ctypes.c_double),
                    ('bandwidth',   ctypes.c_double),
                    ('weighting',   ctypes.c_char*24),
                    ('center_ra',   ctypes.c_double),
                    ('center_dec',  ctypes.c_double),
                    ('center_az',   ctypes.c_double),
                    ('center_alt',  ctypes.c_double),
                    ('asp_filter',  ctypes.c_int),
                    ('asp_atten_1', ctypes.c_int),
                    ('asp_atten_2', ctypes.c_int),
                    ('asp_atten_3', ctypes.c_int),
                    ('payload',     ctypes.c_int),
                    ('extended',    ctypes.c_int)]
    
    _LEGACY_FIELDS = {'asp_atten_3': 'asp_atten_s'}
    
    _TIME_OFFSET_v1 = 4
    _TIME_OFFSET_v2 = _TIME_OFFSET_v1
    _TIME_OFFSET_v3 = _TIME_OFFSET_v2
    _TIME_OFFSET_v4 = _TIME_OFFSET_v3
    _TIME_OFFSET_v5 = _TIME_OFFSET_v4
    _TIME_OFFSET_v6 = _TIME_OFFSET_v5
    
    def __init__(self, filename, mode='r', imager_version='', station='', compression=None):
        """
        Constructs a new OrvilleImageDB.
        
        Optional arguments specify the file mode (must be 'r', 'w', or, 'a';
        defaults to 'r') and strings providing the imager version and the 
        station name, all of which are truncated at 24 bytes.  These optional 
        strings are only relevant when opening a file for writing.
        """
        
        self.name = ''
        self.file = None
        self.curr_int = -1
        self._offsets = []
        self._compress = None
        self._decompress = None
        
        self._FileHeader = self._FileHeader_v6
        self._EntryHeader = self._EntryHeader_v6
        self._TIME_OFFSET = self._TIME_OFFSET_v6
        
        # 'station' is a required keyword
        if mode[0] == 'w' and (station == '' or station == b''):
            raise RuntimeError("'station' is a required keyword for 'mode=w'")
            
        # Sort out 'compression' before we get too far along
        comp, decomp = _get_compression_method(compression)
        self._compress = comp
        self._decompress = decomp
        
        # For read mode, we do not create a new file.  Raise an error if it
        # does not exist, and create an empty OrvilleImageDB object if its length
        # is zero.
        if mode == 'r':
            self._is_new = False
            if not os.path.isfile(filename):
                raise OSError('The specified file, "%s", does not exist.' % filename)
            fileSize = os.path.getsize(filename)
            if fileSize == 0:
                self.version = self._FORMAT_VERSION
                self.header = self._FileHeader()
                self.curr_int = 0
                self.nint = 0
                self.nstokes = 0
                self._offsets = []
                return
                
        # For append mode, check if the file exists and is at least longer
        # than the initial 24 byte version string.  If that's the case, switch
        # to 'r+' mode, since we may need to read and/or write to the header,
        # and some Unix implementations don't allow this with 'a' mode.
        # Otherwise, switch to write mode.
        elif mode == 'a':
            fileSize = os.path.getsize(filename) if os.path.isfile(filename) else 0
            self._is_new = (fileSize <= 24)
            mode = 'w' if self._is_new else 'r+'
            
        # Write mode: pretty straightforward.
        elif mode == 'w':
            self._is_new = True
            
        else:
            raise ValueError("Mode must be 'r', 'w', or 'a'.")
            
        # Now read or create the file header.
        mode += 'b'
        self.name = filename
        self.file = open(filename, mode)
        self._is_outdated = False
        
        if not self._is_new:
            self.version = self.file.read(24).rstrip(b'\x00')
            try:
                self.version = self.version.decode()
            except AttributeError:
                pass
            if self.version != self._FORMAT_VERSION:
                if self.version == 'OrvilleImageDBv001':
                    self._FileHeader = self._FileHeader_v1
                    self._EntryHeader = self._EntryHeader_v1
                    self._TIME_OFFSET = self._TIME_OFFSET_v1
                    self._compress = None
                    self._decompress = None
                elif self.version == 'OrvilleImageDBv002':
                    self._FileHeader = self._FileHeader_v2
                    self._EntryHeader = self._EntryHeader_v2
                    self._TIME_OFFSET = self._TIME_OFFSET_v2
                    self._compress = None
                    self._decompress = None
                elif self.version == 'OrvilleImageDBv003':
                    self._FileHeader = self._FileHeader_v3
                    self._EntryHeader = self._EntryHeader_v3
                    self._TIME_OFFSET = self._TIME_OFFSET_v3
                    self._compress = None
                    self._decompress = None
                elif self.version == 'OrvilleImageDBv004':
                    self._FileHeader = self._FileHeader_v4
                    self._EntryHeader = self._EntryHeader_v4
                    self._TIME_OFFSET = self._TIME_OFFSET_v4
                    self._compress = None
                    self._decompress = None
                elif self.version == 'OrvilleImageDBv005':
                    self._FileHeader = self._FileHeader_v5
                    self._EntryHeader = self._EntryHeader_v5
                    self._TIME_OFFSET = self._TIME_OFFSET_v5
                    self._compress = None
                    self._decompress = None
                else:
                    raise KeyError('The file "%s" does not appear to be a '
                                   'OrvilleImageDB file.  Initial string: "%s"' %
                                   (filename, self.version))
            
            file_header = self._FileHeader()
            
            if mode != 'r' and fileSize <= 24 + ctypes.sizeof(file_header):
                # If the file is too short to have any data in it, close it
                # and start a new one.  This one is probably corrupt.
                self.file.close()
                self._is_new = True
                mode = 'w'
                self.file = open(filename, mode)
            
            else:
                # It looks like we should have a good header, at least ....
                self.header = self._FileHeader()
                self.file.readinto(self.header)
                self.nstokes = len(self.header.stokes_params.split(b','))
                 
                entry_header = self._EntryHeader()
                int_size = ctypes.sizeof(entry_header) \
                            + 4*self.header.nchan*(0 + self.nstokes*self.header.ngrid**2)
                if self.version in ('OrvilleImageDBv006',):
                    # Formats with compression require figuring out how large each integration is
                    comp, decomp = _get_compression_method(self.header.compression)
                    self._compress = comp
                    self._decompress = decomp
                    
                    marker = self.file.tell()
                    end_marker = os.path.getsize(filename)

                    self._offsets = []
                    while self.file.tell() < end_marker:
                        self._offsets.append(self.file.tell())
                        self.file.readinto(entry_header)
                        if entry_header.sync_word != 0xC0DECAFE:
                            raise RuntimeError('The file "%s" appears to be '
                                               'corrupted.' % filename)

                        curr_int_size = int_size + 1*self.header.nchan - ctypes.sizeof(entry_header)
                        if entry_header.payload > 0:
                            curr_int_size = entry_header.payload + 1*self.header.nchan
                        if entry_header.extended > 0:
                            curr_int_size += entry_header.extended
                            
                        self.file.seek(curr_int_size, os.SEEK_CUR)

                    self.nint = len(self._offsets)
                    self.file.seek(marker, os.SEEK_SET)
                    
                else:
                    # Pre-compression versions are easier to validate
                    if self.version in ('OrvilleImageDBv003', 'OrvilleImageDBv004', 'OrvilleImageDBv005'):
                        int_size += 1*self.header.nchan
                        
                    if (fileSize - 24 - ctypes.sizeof(self.header)) % int_size != 0:
                        raise RuntimeError('The file "%s" appears to be '
                                           'corrupted.' % filename)
                    self.nint = \
                        (fileSize - 24 - ctypes.sizeof(self.header)) // int_size
                    data_start = 24 + ctypes.sizeof(self.header)
                    self._offsets = [data_start + int_size * i for i in range(self.nint)]
                    
                if mode == 'r+b':
                    self.file.seek(0, os.SEEK_END)
                    self.curr_int = self.nint
                else:
                    self.curr_int = 0
                    
        if self._is_new:
            # Start preparing a file header, but don't write it until we
            # receive the first image, which will fill in some information
            # (e.g., resolution) that isn't yet available.
            self.version = self._FORMAT_VERSION
            self.header = self._FileHeader()
            try:
                self.header.imager_version = imager_version.encode()
            except AttributeError:
                self.header.imager_version = imager_version
            try:
                self.header.station = station.encode()
            except AttributeError:
                self.header.station = station
            if compression is None:
                compression = 'none'
            try:
                self.header.compression = compression.encode()
            except AttributeError:
                self.header.compression = compression
            self.header.flags = self.FLAG_SORTED     # Sorted until it's not
            self.nint = 0
            
        self.include_mask = (self.version in ('OrvilleImageDBv003', 'OrvilleImageDBv004', 'OrvilleImageDBv005', 'OrvilleImageDBv006'))
        
    def __del__(self):
        if self.file is not None and not self.file.closed:
            self.close()
            
    def close(self):
        """
        Closes the database file.  If the header information is outdated, it
        writes the new file header.
        """
        
        if self.file is None or self.file.closed:  return
        
        if self._is_outdated:
            self.file.seek(24, os.SEEK_SET)
            self.file.write(self.header)
            
        self.file.close()
        self.curr_int = -1
        
    def closed(self):
        return self.file is None or self.file.closed
        
    def getpos(self):
        return self.curr_int
        
    def eof(self):
        return self.curr_int >= self.nint
        
    def seek(self, index):
        if index < 0:
            index += self.nint
        if index < 0 or index >= self.nint:
            raise IndexError('OrvilleImageDB index %d outside of range [0, %d)' %
                             (index, self.nint))
        if self.curr_int != index:
            self.file.seek(self._offsets[index], os.SEEK_SET)
            self.curr_int = index
           
    @property
    def is_compressed(self):
        """
        Whether or not the file uses compression to store the image cubes.
        """
        
        return self._compress is not None
        
    def _check_header(self, stokes_params, ngrid, pixel_size, nchan):
        """
        For new files, adds the given information to the file header and
        writes the header to disk.  For existing files, compares the given
        information to the expected values and raises a ValueError if there's
        a mismatch.
        """
        
        if type(stokes_params) is list:
            stokes_params = ','.join(stokes_params)
        try:
            stokes_params = stokes_params.encode()
        except AttributeError:
            pass
            
        if self._is_new:
            # If this is the file's first image, fill in values of the file
            # header based on the image properties, then write the header.
            self.header.stokes_params = stokes_params
            self.header.ngrid         = ngrid
            self.header.pixel_size    = pixel_size
            self.header.nchan         = nchan
            try:
                self.file.write(struct.pack('<24s', self.version.encode()))
            except AttributeError:
                self.file.write(struct.pack('<24s', self.version))
            self.file.write(self.header)
            self.nstokes = len(self.header.stokes_params.split(b','))
            self._is_new = False
            
        else:
            # Make sure that the Stokes parameters match expectations.
            if stokes_params != self.header.stokes_params:
                raise ValueError(
                    'The Stokes parameters for this image (%s) do not match '
                    'this file\'s parameters (%s).' %
                    (stokes_params, self.header.stokes_params))
                
            # Make sure that the dimensions of the data match expectations.
            if ngrid != self.header.ngrid:
                raise ValueError(
                    'The spatial resolution of this image (%d x %d) does not '
                     'match this file\'s resolution (%d x %d).' %
                    (ngrid, ngrid, self.header.ngrid, self.header.ngrid))
                
            if pixel_size != self.header.pixel_size:
                raise ValueError(
                    'The pixel size of this image (%r deg x %r deg) does not '
                     'match this file\'s resolution (%r deg x %r deg).' %
                    (pixel_size, pixel_size,
                     self.header.pixel_size, self.header.pixel_size))
                
            # Make sure that the size of the images matches expectations.
            if nchan != self.header.nchan:
                raise ValueError(
                    'The channel count for this image (%d) does not '
                    'match this file\'s channel count (%d).'
                    % (nchan, self.header.nchan))
                
    def _update_file_header(self, interval):
        """
        To be called at the end of the add_image functions.  Updates the header
        information to reflect the new data.
        """
        
        self.nint += 1
        
        # Has this image expanded the time range covered by the file?
        if self.header.start_time == 0 or \
           self.header.start_time > interval[0]:
            self.header.start_time = interval[0]
            self._is_outdated = True
            
        if self.header.stop_time < interval[1]:
            self.header.stop_time = interval[1]
            self._is_outdated = True
            
        # If the new image isn't later than all the others, and the file is
        # currently marked as sorted, then remove the sorted flag.
        elif self.header.flags & self.FLAG_SORTED:
            self.header.flags &= ~self.FLAG_SORTED
            self._is_outdated = True
            
    def add_image(self, info, data, mask=None):
        """
        Adds an integration to the database.  Returns the index of the newly
        added image.
        
        Arguments:
        info -- a dictionary with the following keys defined:
            start_time -- MJD UTC at which this integration began
            int_len -- integration length, in days
            fill -- packet fill fraction
            lst -- mean local sidereal time of the observation, in days
            start_freq -- frequency of first channel in the integration, in Hz
            stop_freq -- frequency of last channel in the integration, in Hz
            bandwidth -- bandwidth of each channel in the integrated data, in Hz
            weighting -- string indicating the weighting used during imaging
            center_ra -- RA of the image phase center, in degrees
            center_dec -- Declination of image phase center, in degrees
            center_az -- azimuth of the image phase center, in degrees
            center_alt -- altitude of image phase center, in degrees
            asp_filter -- (optional) ASP filter code (0=split, 1=full, ...)
            asp_atten_1 -- (optional) ASP first attenuator setting
            asp_atten_2 -- (optional) ASP second attenuator setting
            asp_atten_3 -- (optional) ASP third attenuator setting
            extended_attributes -- (optional) Dictionary containing JSON-able
                                   extra attributes to include
            pixel_size -- Real-world size of a pixel, in degrees
            stokes_params -- a list or comma-delimited string of Stokes params
        data -- a 4D float array of image data indexed as [chan, stokes, x, y]
        mask -- (optional) a 1D uint8 of frequency masking flags as [chan,]
        """
        
        assert(data.shape[2] == data.shape[3])
        if isinstance(data, np.ma.MaskedArray):
            if self.include_mask:
                if mask is None:
                    mask = data.mask[:,0,0,0]
                assert(mask.size == data.shape[0])
            data = data.data
        else:
            if self.include_mask:
                if mask is None:
                    mask = np.zeros(data.shape[0], dtype=np.uint8)
                assert(mask.size == data.shape[0])
        self._check_header(info['stokes_params'], data.shape[2], 
                           info['pixel_size'], data.shape[0])
        
        # Write it out.
        entry_offset = self.file.tell()
        entry_header = self._EntryHeader()
        entry_header.sync_word = 0xC0DECAFE
        for key in ('start_time', 'int_len', 'fill', 'lst', 'start_freq', 'stop_freq',
                    'bandwidth', 'weighting', 'center_ra', 'center_dec', 'center_az', 'center_alt',
                    'asp_filter', 'asp_atten_1', 'asp_atten_2', 'asp_atten_3'):
            if key in ('weighting', 'fill', 'center_az', 'center_alt', 'asp_atten_3') and self.version != self._FORMAT_VERSION:
                continue
            if key == 'weighting':
                if self.version != self._FORMAT_VERSION:
                    continue
                elif key not in info:
                    info[key] = b'natural'
                else:
                    try:
                        info[key] = info[key].encode()
                    except AttributeError:
                        # Already bytes
                        pass
            elif key.startswith('asp_'):
                if self.version != self._FORMAT_VERSION:
                    continue
                elif key not in info:
                    legacy = self._LEGACY_FIELDS.get(key)
                    info[key] = info.get(legacy, -1) if legacy else -1
            setattr(entry_header, key, info[key])
            
        extended = None
        if 'extended_attributes' in info:
            extended = json.dumps(info['extended_attributes']).encode()
            setattr(entry_header, 'extended', len(extended))
        else:
            setattr(entry_header, 'extended', -1)
        if self._compress is not None:
            cdata = self._compress(data)
            setattr(entry_header, 'payload', len(cdata))
            self.file.write(entry_header)
            self.file.write(cdata)
        else:
            setattr(entry_header, 'payload', -1)
            self.file.write(entry_header)
            data.astype('<f4').tofile(self.file)
        if self.include_mask:
            mask.astype('u1').tofile(self.file)
        if extended:
            self.file.write(extended)
        self.file.flush()
        
        interval = [info['start_time'], info['start_time'] + info['int_len']]
        self._update_file_header(interval)
        
        self._offsets.append(entry_offset)
        return self.nint - 1
        
    def read_image(self):
        """
        Reads an integration from the database.
        
        Returns a 2-tuple containing:
        info -- a dictionary with the following keys defined:
            start_time -- MJD UTC at which this integration began
            int_len -- integration length, in days
            fill -- packet fill fraction, if available
            lst -- mean local sidereal time of the observation, in days
            start_freq -- frequency of first channel in the integration, in Hz
            stop_freq -- frequency of last channel in the integration, in Hz
            bandwidth -- bandwidth of each channel in the integrated data, in Hz
            weighting - image weighting used
            center_ra -- RA of the image phase center, in degrees
            center_dec -- Declination of image phase center, in degrees
            center_az -- azimuth of the image phase center, in degrees
            center_alt -- altitude of image phase center, in degrees
            asp_filter -- ASP filter code (0=split, 1=full, ...)
            asp_atten_1 -- ASP first attenuator setting
            asp_atten_2 -- ASP second attenuator setting
            asp_atten_3 -- ASP third attenuator setting
            extended_attributes -- (optional) Dictionary containing JSON-able
                                   extra attributes to include
            pixel_size -- Real-world size of a pixel, in degrees
            stokes_params -- a list or comma-delimited string of Stokes params
        data -- a 4D float array of image data indexed as [chan, stokes, x, y]
        """
        
        if self.curr_int >= self.nint:
            raise IOError("end of file reached")
            
        entry_header = self._EntryHeader()
        self.file.readinto(entry_header)
        if entry_header.sync_word != 0xC0DECAFE:
            raise RuntimeError("Database corrupted")
        info = {}
        for key in ('stokes_params', 'pixel_size', 'ngrid', 'station'):
            info[key] = getattr(self.header, key, None)
        for key in ('start_time', 'int_len', 'fill', 'lst', 'start_freq', 'stop_freq',
                    'bandwidth', 'weighting', 'center_ra', 'center_dec', 'center_az', 'center_alt',
                    'asp_filter', 'asp_atten_1', 'asp_atten_2', 'asp_atten_3', 'payload', 'extended'):
            info[key] = getattr(entry_header, key, None)
            if info[key] is None and key in self._LEGACY_FIELDS:
                info[key] = getattr(entry_header, self._LEGACY_FIELDS[key], None)
            
            if key == 'weighting' and info[key] is None:
                    info[key] = b'natural'
            elif key.startswith('asp_') and info[key] is None:
                info[key] = -1
            elif key in ('payload', 'extended') and info[key] is None:
                info[key] = -1
                
        nchan, nstokes, ngrid = self.header.nchan, self.nstokes, self.header.ngrid
        if info['payload'] > 0:
            cdata = self.file.read(info['payload'])
            data = self._decompress(cdata)
        else:
            data = np.fromfile(self.file, '<f4', nchan*nstokes*ngrid*ngrid)
            data = data.reshape(nchan, nstokes, ngrid, ngrid)
        del info['payload']
        if self.include_mask:
            mask = np.fromfile(self.file, 'u1', nchan)
            reshaped_mask = np.full(data.shape, False, dtype=bool) # Create Bool array filled with False values
            reshaped_mask[np.argwhere(mask),...] = True # Propagate True across rows of flagged channels
            data = np.ma.masked_array(data, reshaped_mask, dtype=data.dtype) # Create masked array
        if info['extended'] > 0:
            extended = self.file.read(info['extended'])
            info['extended_attributes'] = json.loads(extended)
        del info['extended']
         
        self.curr_int += 1
        return info, data
        
    def read_all(self):
        """
        Reads all integrations from the database.
        
        Returns a 2-tuple containing:
        hdr_list -- a list of dictionaries with the following keys defined:
            start_time -- MJD UTC at which this integration began
            int_len -- integration length, in days
            fill -- packet fill fraction, if available
            lst -- mean local sidereal time of the observation, in days
            start_freq -- frequency of first channel in the integration, in Hz
            stop_freq -- frequency of last channel in the integration, in Hz
            bandwidth -- bandwidth of each channel in the integrated data, in Hz
            weighting - image weighting used
            center_ra -- RA of the image phase center, in degrees
            center_dec -- Declination of image phase center, in degrees
            center_az -- azimuth of the image phase center, in degrees
            center_alt -- altitude of image phase center, in degrees
            asp_filter -- ASP filter code (0=split, 1=full, ...)
            asp_atten_1 -- ASP first attenuator setting
            asp_atten_2 -- ASP second attenuator setting
            asp_atten_3 -- ASP third attenuator setting
            extended_attributes -- (optional) Dictionary containing JSON-able
                                   extra attributes to include
            pixel_size -- Real-world size of a pixel, in degrees
            stokes_params -- a list or comma-delimited string of Stokes params
        data_all -- a 5D masked float32 array of image data indexed as 
            [integration, chan, stokes, x, y]
        """
        
        self.seek(0)
        nchan, nstokes, ngrid = self.header.nchan, self.nstokes, self.header.ngrid
        data_all = np.ma.zeros((self.nint, nchan, nstokes, ngrid, ngrid), dtype=np.float32)
        hdr_list = []
        while self.curr_int < self.nint:
            hdr, data = self.read_image()
            hdr_list.append(hdr)
            data_all[self.curr_int-1] = data
        return hdr_list, data_all
        
    @staticmethod
    def sort(filename):
        """
        Sorts the integrations in a DB file to be time-ordered.
        """
        
        is_interactive = sys.__stdin__.isatty()
        
        # Open the input database.  If it's already sorted, stop.
        inDB = OrvilleImageDB(filename, 'r')
        if inDB.header.flags & OrvilleImageDB.FLAG_SORTED:
            inDB.close()
            return
            
        # First pass: read just the start time from each integration
        # using the offset table.
        file_size = os.path.getsize(filename)
        times = np.empty(inDB.nint, dtype=np.float64)
        for i in range(inDB.nint):
            inDB.file.seek(inDB._offsets[i] + inDB._TIME_OFFSET, os.SEEK_SET)
            times[i] = struct.unpack('<d', inDB.file.read(8))[0]
        intOrder = times.argsort()
        
        # Second pass: write the sorted file one integration at a time,
        # seeking to each integration in sorted order.
        inDB.header.flags |= OrvilleImageDB.FLAG_SORTED
        
        with tempfile.NamedTemporaryFile(mode='wb', prefix='orville-', suffix='.oims') as outFile:
            outVersion = inDB.version
            try:
                outFile.write(struct.pack('<24s', outVersion.encode()))
            except AttributeError:
                outFile.write(struct.pack('<24s', outVersion))
            outFile.write(inDB.header)
            outFile.flush()
            
            pbar = ProgressBarPlus(max=inDB.nint)
            for iOut in range(inDB.nint):
                idx = intOrder[iOut]
                inDB.file.seek(inDB._offsets[idx], os.SEEK_SET)
                if idx + 1 < inDB.nint:
                    blob_size = inDB._offsets[idx + 1] - inDB._offsets[idx]
                else:
                    blob_size = file_size - inDB._offsets[idx]
                outFile.write(inDB.file.read(blob_size))
                pbar.inc(1)
                
                if is_interactive:
                    sys.stdout.write(pbar.show()+'\r')
                    sys.stdout.flush()
                    
            outFile.flush()
            if is_interactive:
                sys.stdout.write(pbar.show()+'\n')
                sys.stdout.flush()
                
            # Overwrite the original file
            shutil.copy(outFile.name, filename)
            
        inDB.close()
        
    @staticmethod
    def compress(filename, method='fpzip'):
        """
        Compresses the integrations in a DB file using fpzip.
        """
        
        is_interactive = sys.__stdin__.isatty()
        
        # Open the input database.  If it's already compressed, stop.
        inDB = OrvilleImageDB(filename, 'r')
        
        # Write the compressed file.  Note that we write it using the most recent
        # header version, which may differ from the version of the input file.
        # After writing the updated file header, loop through the intervals
        # and compress from the input data to the output file.
        with tempfile.NamedTemporaryFile(mode='wb', prefix='orville-', suffix='.oims') as outFile:
            outname = outFile.name
            outDB = OrvilleImageDB(outname, mode='w',
                                   imager_version=inDB.header.imager_version,
                                   station=inDB.header.station,
                                   compression=method)
            
            pbar = ProgressBarPlus(max=inDB.nint)
            for (header,image) in inDB:
                if isinstance(image, np.ma.MaskedArray):
                    mask = image.mask.any(axis=(1,2,3)).astype(np.uint8)
                    outDB.add_image(header, image.data, mask=mask)
                else:
                    outDB.add_image(header, image)
                pbar.inc(1)
                
                if is_interactive:
                    sys.stdout.write(pbar.show()+'\r')
                    sys.stdout.flush()
                    
            outDB.close()
            if is_interactive:
                sys.stdout.write(pbar.show()+'\n')
                sys.stdout.flush()
                
            # Overwrite the original file
            origsize = os.path.getsize(filename)
            compsize = os.path.getsize(outFile.name)
            shutil.copy(outFile.name, filename)
            if is_interactive:
                print("%i B in, %i B out (compressed %.1f%%)" % (origsize, compsize, 100.0*(origsize-compsize)/origsize))
                
        inDB.close()
            
    # Implement some built-ins to make reading images more "Pythonic" ...
    def __enter__(self):
        return self
        
    def __exit__(self, type, value, tb):
        self.close()
        
    def __len__(self):
        return self.nint
        
    def __getitem__(self, index):
        if index >= self.nint:
            raise IndexError("image index out of range")
            
        self.seek(index)
        return self.read_image()
        
    def __iter__(self):
        return self
        
    def __next__(self):
        return self.next()
        
    def next(self):
        if self.curr_int >= self.nint:
            raise StopIteration
        else:
            return self.read_image()
