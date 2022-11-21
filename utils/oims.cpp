#include <iostream>
#include <stdexcept>

#include "oims.hpp"

OrvilleImageDB::OrvilleImageDB(std::string filename) {
  _filename = filename;
  _fh.open(filename, std::ios::in|std::ios::binary);
  if( !_fh.good() ) {
    throw(std::runtime_error("Failed to open file"));
  }
  
  std::string version_string;
  _fh.read(reinterpret_cast<char*>(&_fileh), sizeof(_fileh));
  version_string = std::string(_fileh.format_version);
  if( version_string == std::string("OrvilleImageDBv001") ) {
    _version = 1;
  } else if( version_string == std::string("OrvilleImageDBv002") ) {
    _version = 2;
  } else if( version_string == std::string("OrvilleImageDBv003") ) {
    _version = 3;
  } else if( version_string == std::string("OrvilleImageDBv004") ) {
    _version = 4;
  } else {
    throw(std::runtime_error("Unknown file format"));
  }
  
  std::string stokes_names = std::string(_fileh.stokes_params);
  size_t sep = stokes_names.find(",", 0);
  while( sep < stokes_names.size() ) {
    _stokes.push_back(stokes_names.substr(0, sep));
    stokes_names = stokes_names.substr(sep+1, stokes_names.size()-sep-1);
  }
  _stokes.push_back(stokes_names);
  
  _nstokes = _stokes.size();
  
  uint64_t marker, size;
  marker= _fh.tellg();
  _fh.seekg (0, std::ios::end);
  size = _fh.tellg();
  _fh.seekg(marker, std::ios::beg);
  
  _frame_size = 0;
  switch(_version) {
    case 1: _frame_size += sizeof(OrvilleFrameHeaderV1); break;
    case 2: _frame_size += sizeof(OrvilleFrameHeaderV2); break;
    default: _frame_size += sizeof(OrvilleFrameHeaderV3) + sizeof(uint8_t)*_fileh.nchan;
  }
  _frame_size += sizeof(float)*_fileh.nchan*(0 + _nstokes*_fileh.ngrid*_fileh.ngrid);
  
  _frame_count = (size - sizeof(_fileh)) / _frame_size;
  if( (size - sizeof(_fileh)) % _frame_size != 0 ) {
    throw(std::runtime_error("File appears corrupted"));
  }
  
  _frame_idx = 0;
}


OrvilleImage* OrvilleImageDB::read() {
  if( _frame_idx >= _frame_count ) {
    return NULL;
  }
  
  switch(_version) {
    case 1: {
      OrvilleFrameHeaderV1 _tmp;
      _fh.read(reinterpret_cast<char*>(&_tmp), sizeof(_tmp));
      _frameh.sync_word  = _tmp.sync_word;
      _frameh.start_time = _tmp.start_time;
      _frameh.int_len    = _tmp.int_len;
      _frameh.lst        = _tmp.lst;
      _frameh.start_freq = _tmp.start_freq;
      _frameh.stop_freq  = _tmp.stop_freq;
      _frameh.bandwidth  = _tmp.bandwidth;
      _frameh.center_ra  = _tmp.center_ra;
      _frameh.center_dec = _tmp.center_dec;
      break;
    }
    case 2: {
      OrvilleFrameHeaderV2 _tmp;
      _fh.read(reinterpret_cast<char*>(&_tmp), sizeof(_tmp));
      _frameh.sync_word  = _tmp.sync_word;
      _frameh.start_time = _tmp.start_time;
      _frameh.int_len    = _tmp.int_len;
      _frameh.fill       = _tmp.fill;
      _frameh.lst        = _tmp.lst;
      _frameh.start_freq = _tmp.start_freq;
      _frameh.stop_freq  = _tmp.stop_freq;
      _frameh.bandwidth  = _tmp.bandwidth;
      _frameh.center_ra  = _tmp.center_ra;
      _frameh.center_dec = _tmp.center_dec;
      _frameh.center_az  = _tmp.center_az;
      _frameh.center_alt = _tmp.center_alt;
      break;
    }
    default: _fh.read(reinterpret_cast<char*>(&_frameh), sizeof(_frameh));
  }
  
  OrvilleImage *frame = new OrvilleImage();
  frame->_station = std::string(_fileh.station);
  frame->_nchan = _fileh.nchan;
  frame->_nstokes = _nstokes;
  frame->_stokes = _stokes;
  frame->_ngrid = _fileh.ngrid;
  frame->_pixel_size = _fileh.pixel_size;

  frame->_hdr.sync_word   = _frameh.sync_word;
  frame->_hdr.start_time  = _frameh.start_time;
  frame->_hdr.int_len     = _frameh.int_len;
  frame->_hdr.fill        = _frameh.fill;
  frame->_hdr.lst         = _frameh.lst;
  frame->_hdr.start_freq  = _frameh.start_freq;
  frame->_hdr.stop_freq   = _frameh.stop_freq;
  frame->_hdr.bandwidth   = _frameh.bandwidth;
  frame->_hdr.center_ra   = _frameh.center_ra;
  frame->_hdr.center_dec  = _frameh.center_dec;
  frame->_hdr.center_az   = _frameh.center_az;
  frame->_hdr.center_alt  = _frameh.center_alt;
  frame->_hdr.asp_filter  = _frameh.asp_filter;
  frame->_hdr.asp_atten_1 = _frameh.asp_atten_1;
  frame->_hdr.asp_atten_2 = _frameh.asp_atten_2;
  frame->_hdr.asp_atten_s = _frameh.asp_atten_s;

  frame->_image = (float*) calloc(sizeof(float), _fileh.nchan*(0 + _nstokes*_fileh.ngrid*_fileh.ngrid));
  frame->_mask = (uint8_t*) calloc(sizeof(uint8_t), _fileh.nchan*(0 + _nstokes*_fileh.ngrid*_fileh.ngrid)); 
  _fh.read(reinterpret_cast<char*>(frame->_image), sizeof(float)*_fileh.nchan*(0 + _nstokes*_fileh.ngrid*_fileh.ngrid));
  if( _version >= 3) {
    _fh.read(reinterpret_cast<char*>(frame->_mask), sizeof(uint8_t)*_fileh.nchan*(0 + _nstokes*_fileh.ngrid*_fileh.ngrid));
  }
  
  _frame_idx++;
  return frame;
}
