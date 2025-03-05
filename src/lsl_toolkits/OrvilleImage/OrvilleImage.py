import h5py
import numpy as np
import os
import time
import json
from functools import lru_cache
from typing import Dict, Any, Optional, Tuple, List, TypeVar, Union


K = TypeVar('K')
V = TypeVar('V')

class HeaderContainer(Dict[K, V]):
    """
    Sub-class of dict that supports access of keys as attributes.
    """
    
    def __getattr__(self, key: K) -> V:
        """
        Support for attribute-style access: header.key
        """
        
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Header has no attribute '{key}'")


class OrvilleImageHDF5:
    """
    HDF5-based implementation of OrvilleImageDB.
    
    Provides similar functionality to the original OrvilleImageDB but with the 
    advantages of HDF5 including better crash recovery, compression, and parallel access.
    """
    
    _FORMAT_VERSION = 'OrvilleImageDBv006'
    
    _REQUIRED_IMAGE_METADATA = ['start_time', 'int_len',
                                'start_freq', 'stop_freq', 'bandwidth',
                                'stokes_params',
                                'pixel_size', 'center_ra', 'center_dec'
                               ]
    
    def __init__(self, filename: str, mode: str='r',
                       imager_version: str='', station: str='',
                       time_format: str='mjd', time_scale: str='utc',
                       compression: Optional[Union[str,int]]=None):
        """
        Constructs a new OrvilleImageHDF5.
        
        Args:
            filename: Path to the HDF5 file
            mode: Access mode ('r', 'w', 'a')
            imager_version: String providing the imager version (for writing)
            station: Station name (for writing)
            time_format: time format (jd, mjd, etc.; for writing)
            time_scale: time scale (utc, tai etc.; for writing)
            compression: HDF5 compression method (for writing)
        """
        
        self.name = filename
        self._is_new = False
        
        # Handle different modes
        if mode == 'r':
            if not os.path.isfile(filename):
                raise OSError(f'The specified file, "{filename}", does not exist.')
            
            self.h5 = h5py.File(filename, 'r')
            self._setup_existing_file()
            
        elif mode == 'a':
            if os.path.isfile(filename):
                self.h5 = h5py.File(filename, 'a', libver='latest')
                self._setup_existing_file()
            else:
                self._is_new = True
                self.h5 = h5py.File(filename, 'w', libver='latest')
                self._setup_new_file(imager_version, station, time_format, time_scale)
                
        elif mode == 'w':
            self._is_new = True
            self.h5 = h5py.File(filename, 'w', libver='latest')
            self._setup_new_file(imager_version, station, time_format, time_scale)
            
        else:
            raise ValueError("Mode must be 'r', 'w', or 'a'.")
            
        # Enable single writer, multiple reader (SWMR) mode if available
        if mode != 'r' and hasattr(self.h5, 'swmr_mode'):
            self.h5.swmr_mode = True
            self.compression = compression
            
    def _setup_new_file(self, imager_version: str, station: str,
                              time_format: str, time_scale: str):
        """
        Set up a new HDF5 file with the appropriate structure.
        """
        
        # Create root attributes
        self.h5.attrs['format_version'] = self._FORMAT_VERSION
        self.h5.attrs['creation_time'] = time.time()
        
        # Create groups for header info and images
        self.h5.create_group('header')
        self.h5.create_group('images')
        
        # Initialize header with empty/default values
        header = self.h5['header']
        header.attrs['imager_version'] = imager_version
        header.attrs['station'] = station
        header.attrs['stokes_params'] = ''
        header.attrs['ngrid'] = 0
        header.attrs['pixel_size'] = 0.0
        header.attrs['nchan'] = 0
        header.attrs['flags'] = 0
        header.attrs['start_time'] = 0.0
        header.attrs['stop_time'] = 0.0
        header.attrs['time_format'] = time_format
        header.attrs['time_scale'] = time_scale
        
        # Properties
        self._header = header
        self.nint = 0
        self.nstokes = 0
        
    def _setup_existing_file(self):
        """
        Set up variables based on an existing HDF5 file.
        """
        
        # Check format version
        if 'format_version' not in self.h5.attrs:
            raise ValueError(f"File {self.name} is not a valid OrvilleImageDB_HDF5 file.")
            
        version = self.h5.attrs['format_version']
        if version != self._FORMAT_VERSION:
            # Could handle version differences here if needed
            print(f"Warning: File version {version} doesn't match current version {self._FORMAT_VERSION}")
        
        # Set up header access
        self._header = self.h5['header']
        
        # Count number of integrations and stokes parameters
        self.nint = len(self.h5['images'])
        if 'stokes_params' in self._header.attrs:
            stokes_str = self._header.attrs['stokes_params']
            self.nstokes = len(stokes_str.split(','))
            
    @property
    def header(self) -> HeaderContainer[str, Any]:
        """
        The file header as a dictionary.
        """
        
        h = HeaderContainer()
        for k,v in self._header.attrs.items():
            h[k] = v
        return h
        
    def _verify_metadata(self, info: Dict[str, Any]):
        """
        Verify that the image metadata contains everything that it needs to.
        """
        
        missing = []
        for key in self._REQUIRED_IMAGE_METADATA:
            if key not in info:
                missing.append(key)
        if missing:
            raise ValueError(f"Metadata missing required keywords: {', '.join(missing)}")
            
    def _check_and_update_header(self, info: Dict[str, Any],
                                       data_shape: Tuple[int, int, int, int]):
        """
        Set up header information based on the first image.
        """
        
        # Extract and convert stokes_params if needed
        stokes_params = info['stokes_params']
        if not isinstance(stokes_params, str):
            stokes_params = ','.join(stokes_params)
        
        # Update header attributes
        self._header.attrs['stokes_params'] = stokes_params
        self._header.attrs['ngrid'] = data_shape[2]
        self._header.attrs['pixel_size'] = info['pixel_size']
        self._header.attrs['nchan'] = data_shape[0]
        self._header.attrs['flags'] = 1  # Sorted by default
        
        # Set time range
        start_time = info['start_time']
        stop_time = start_time + info['int_len']
        self._header.attrs['start_time'] = start_time
        self._header.attrs['stop_time'] = stop_time
        
        # Set stokes count
        self.nstokes = len(stokes_params.split(','))
    
    def _verify_data_compatibility(self, info: Dict[str, Any],
                                         data_shape: Tuple[int, int, int, int]):
        """
        Verify that new data is compatible with the existing images in the
        file.
        """
        
        if data_shape[0] != self._header.attrs['nchan']:
            raise ValueError(f"Channel count mismatch: got {data_shape[0]}, expected {self._header.attrs['nchan']}")
            
        if data_shape[1] != self.nstokes:
            raise ValueError(f"Stokes parameter count mismatch: got {data_shape[1]}, expected {self.nstokes}")
            
        if data_shape[2] != self._header.attrs['ngrid']:
            raise ValueError(f"Grid size mismatch: got {data_shape[2]}, expected {self._header.attrs['ngrid']}")
            
        stokes_params = info['stokes_params']
        if not isinstance(stokes_params, str):
            stokes_params = ','.join(stokes_params)
            
        if stokes_params != self._header.attrs['stokes_params']:
            raise ValueError(f"Stokes parameters mismatch: got {stokes_params}, expected {self._header.attrs['stokes_params']}")
            
        if info['pixel_size'] != self._header.attrs['pixel_size']:
            raise ValueError(f"Pixel size mismatch: got {info['pixel_size']}, expected {self._header.attrs['pixel_size']}")
            
    def _update_time_range(self, info: Dict[str, Any]):
        """
        Update the database time range based on new image.
        """
        
        start_time = info['start_time']
        stop_time = start_time + info['int_len']
        
        # Update global time range if needed
        if start_time < self._header.attrs['start_time'] \
           or self._header.attrs['start_time'] == 0:
            self._header.attrs['start_time'] = start_time
            
        if stop_time > self._header.attrs['stop_time']:
            self._header.attrs['stop_time'] = stop_time
            
        # Check if still sorted
        if self.nint > 0 and self._header.attrs['flags'] & 1:
            last_int = self.h5['images'][f'int_{self.nint-1}']
            if start_time < last_int.attrs['start_time']:
                # No longer time-sorted
                self._header.attrs['flags'] &= ~1
                
    @staticmethod
    @lru_cache(maxsize=16)
    def _get_chunk_size(shape: Tuple, item_size: int,
                        min_bytes: int=100*1024,
                        max_bytes: int=1*1024*1024) -> Union[Tuple, bool]:
        """
        Calculate an appropriate chunk size for HDF5 dataset based on shape and data type.
        
        Args:
            shape: Tuple representing the array shape
            dtype: NumPy data type of the array
            min_bytes: Minimum target chunk size in bytes (default: 100 KB)
            max_bytes: Maximum target chunk size in bytes (default: 1 MB)
            
        Returns:
            Either a tuple of chunk dimensions or True if no good chunking strategy is found
        """
        
        # Get the total size of the data
        data_size = np.prod(shape)*item_size
        if data_size < min_bytes:
            return True
            
        # Create the chunking variables and figure out how we can best divide
        # each access by factors of 2, 3, 5, or 7
        chunk_shape = list(shape)
        chunk_size = data_size
        chunk_factors = {}
        for i in range(len(chunk_shape)):
            temp = chunk_shape[i]
            chunk_factors[i] = []
            for j in (2, 3, 5, 7):
                while temp % j == 0:
                    chunk_factors[i].append(j)
                    temp //= j
                    
        # While we are larger than the chunk size, work on making smaller
        # chunks
        while chunk_size > max_bytes and any([s for s in chunk_shape]):
            ## Look for an axis that we can divide
            dim_to_reduce = None
            for dim in range(4):
                if chunk_shape[dim] > 1 and len(chunk_factors[dim]):
                    dim_to_reduce = dim
                    break
                    
            ## No axis, found, continue
            if dim_to_reduce is None:
                break
                
            ## Yep, divide that axis
            chunk_shape[dim_to_reduce] = max(1, chunk_shape[dim_to_reduce]//chunk_factors[dim_to_reduce].pop(0))
            chunk_size = np.prod(chunk_shape)*item_size
            
        # Make sure we are still above the mimum chunk size
        if chunk_size < min_bytes:
            return True
            
        return tuple(chunk_shape)
        
    def add_image(self, info: Dict[str, Any], data: np.ndarray,
                        mask: Optional[np.ndarray]=None) -> int:
        """
        Adds an integration to the database.
        
        Args:
            info: Dictionary with metadata
            data: A 4D float array of image data indexed as [chan, stokes, x, y]
            mask: Optional 1D uint8 of frequency masking flags as [chan,]
            
        Returns:
            The index of the newly added image
        """
        
        # Basic imput verification
        self._verify_metadata(info)
        if len(data.shape) != 4:
            raise RuntimeError(f"Expected 4D not {len(data.shape)}D data")
        if data.shape[2] != data.shape[3]:
            raise RuntimeError(f"Image data is not square: {data.shape[2]} != {data.shape[3]}")
            
        # If we have a mask, make sure it's the correct size
        if mask is not None:
            if mask.size != data.shape[0]:
                raise RuntimeError(f"Mismatch between channel mask size and the data shape: {mask.size} != {data.shape[0]}")
                
        # On first image, set up header info
        if self._is_new:
            self._check_and_update_header(info, data.shape)
            self._is_new = False
        else:
            # Verify data matches existing database parameters
            self._verify_data_compatibility(info, data.shape)
            
        # Create a new image group
        img_group = self.h5['images'].create_group(f'int_{self.nint}')
        
        # Store image metadata
        for key, value in info.items():
            if key in ('stokes_params', 'pixel_size'):
                continue  # These are global attributes
            if isinstance(value, (list, tuple, dict)):
                value = json.dumps(value)
            img_group.attrs[key] = value
            
        # Store the image data with optional compression
        chunks = self._get_chunk_size(data.shape, data.dtype.itemsize)
        d = img_group.create_dataset('data', data=data, chunks=chunks,
                                     compression=self.compression)
        #d.attrs['axis0'] = 'channel'
        #d.attrs['axis1'] = 'stokes'
        #d.attrs['axis2'] = 'x'
        #d.attrs['axis3'] = 'y'
        
        # Store mask if provided
        if mask is not None:
            m = img_group.create_dataset('mask', data=mask,
                                         compression=self.compression)
            #m.attrs['axis0'] = 'channel'
            
        # Update header time range if needed
        self._update_time_range(info)
        
        # Increment image count
        self.nint += 1
        
        # Make sure to flush to disk
        self.h5.flush()
        
        return self.nint - 1
        
    def read_image(self, idx: int) -> Tuple[HeaderContainer[str, Any], np.ndarray]:
        """
        Read in the metadata and image at the specified index.
        """
        
        if idx < 0:
            idx = self.nint - idx
        elif idx >= self.nint:
            raise IndexError("Requested index is out of range")
            
        img_group = self.h5['images'][f"int_{idx}"]
        info = HeaderContainer({'stokes_params': self._header.attrs['stokes_params'],
                                'pixel_size': self._header.attrs['pixel_size'],
                                'time_format': self._header.attrs['time_format'],
                                'time_scale': self._header.attrs['time_scale']
                               })
        for key in img_group.attrs:
            info[key] = img_group.attrs[key]
            if isinstance(info[key], bytes):
                info[key] = json.loads(info[key])
                
        data = img_group['data'][...]
        if 'mask' in img_group:
            full_mask = np.zeros(data.shape, dtype=bool)
            full_mask[np.argwhere(img_group['mask']),...] = True
            data = np.ma.array(data, mask=full_mask, dtype=data.dtype)
            
        return info, data
        
    def read_all(self) -> Tuple[List[HeaderContainer[str, Any]], np.ndarray]:
        """
        Read in all metadata and images at once and return them.
        """
        
        metadata_list = []
        data_array = None
        for i,(metadata,data) in enumerate(self):
            metadata_list.append(metadata)
            if data_array is None:
                data_array = np.ma.empty((self.nint,)+data.shape, dtype=data.dtype)
            data_array[i] = data
            
        return metadata_list, data_array
        
    def close(self):
        """Close the HDF5 file."""
        if hasattr(self, 'h5') and self.h5:
            self.h5.close()
            self.h5 = None
            
    def __del__(self):
        """Ensure file is closed on object deletion."""
        self.close()
        
    def __enter__(self) -> 'OrvilleImageHDF5':
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
        
    def __len__(self) -> int:
        return self.nint
        
    def __getitem__(self, idx: int):
        return self.read_image(idx)
        
    def append(self, metadata: Dict[str, Any], data: np.ndarray,
                     mask: Optional[np.ndarray]=None):
        self.add_image(metadata, data, mask=mask)
