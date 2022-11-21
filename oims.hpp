#ifndef __INCLUDE_OIMS_HPP
#define __INCLUDE_OIMS_HPP

#include <fstream>
#include <string>
#include <list>
#include <utility>

typedef struct __attribute__((packed)) {
  char     format_version[24];
  char     imager_version[24];
  char     station[24];
  char     stokes_params[24];
  int32_t  ngrid;
  double   pixel_size;
  int32_t  nchan;
  uint32_t flags;
  double   start_time;
  double   stop_time;
} OrvilleFileHeader;

typedef struct __attribute__((packed)) {
  uint32_t sync_word;
  double   start_time;
  double   int_len;
  double   lst;
  double   start_freq;
  double   stop_freq;
  double   bandwidth;
  double   center_ra;
  double   center_dec;
} OrvilleFrameHeaderV1;

typedef struct __attribute__((packed)) {
  uint32_t sync_word;
  double   start_time;
  double   int_len;
  double   fill;
  double   lst;
  double   start_freq;
  double   stop_freq;
  double   bandwidth;
  double   center_ra;
  double   center_dec;
  double   center_az;
  double   center_alt;
} OrvilleFrameHeaderV2;

typedef struct __attribute__((packed)) {
  uint32_t sync_word;
  double   start_time;
  double   int_len;
  double   fill;
  double   lst;
  double   start_freq;
  double   stop_freq;
  double   bandwidth;
  double   center_ra;
  double   center_dec;
  double   center_az;
  double   center_alt;
  int32_t  asp_filter;
  int32_t  asp_atten_1;
  int32_t  asp_atten_2;
  int32_t  asp_atten_s;
} OrvilleFrameHeaderV3;

class OrvilleImage {
friend class OrvilleImageDB;
private:
  std::string _station;
  int32_t _nchan;
  int32_t _nstokes;
  std::list<std::string> _stokes;
  int32_t _ngrid;
  double _pixel_size;
  
  OrvilleFrameHeaderV3 _hdr;
  float*   _image;
  uint8_t* _mask;
  
public:
  OrvilleImage(): _nchan(0), _nstokes(0), _ngrid(0), _image(NULL), _mask(NULL) {}
  ~OrvilleImage() {
    if( _image == NULL ) {
      ::free(_image);
    }
    if( _mask == NULL ) {
      ::free(_mask);
    }
  }
  inline std::string get_station()  { return _station;         }
  inline int32_t  get_nchan()       { return _nchan;           }
  inline int32_t  get_nstokes()     { return _nstokes;         }
  inline std::list<std::string> get_stokes() {
    return _stokes;
  }
  inline int32_t  get_ngrid()        { return _ngrid;          }
  inline double   get_pixel_size()   { return _pixel_size;     }
  inline double   get_start_time()   { return _hdr.start_time; }
  inline double   get_int_len()      { return _hdr.int_len;    }
  inline double   get_fill()         { return _hdr.fill;       }
  inline double   get_lst()          { return _hdr.lst;        }
  inline double   get_start_freq()   { return _hdr.start_freq; }
  inline double   get_stop_freq()    { return _hdr.stop_freq;  }
  inline double   get_bandwidth()    { return _hdr.bandwidth;  }
  inline std::pair<double,double> get_center_equ() {
    return std::make_pair(_hdr.center_ra, _hdr.center_dec);
  }
  inline std::pair<double,double> get_center_top() {
    return std::make_pair(_hdr.center_az, _hdr.center_alt);
  }
  inline int32_t  get_asp_filter()   { return _hdr.asp_filter; }
  inline float*   get_image()        { return _image;          }
  inline uint8_t* get_mask()         { return _mask;           }
};

class OrvilleImageDB {
private:
  std::string   _filename;
  std::ifstream _fh;
  int32_t       _version;
  
  int32_t                _nstokes;
  std::list<std::string> _stokes;
  
  OrvilleFileHeader    _fileh;
  OrvilleFrameHeaderV3 _frameh;
  
  uint32_t _frame_size;
  uint32_t _frame_count;
  uint32_t _frame_idx;
  
public:
  OrvilleImageDB(std::string filename);
  ~OrvilleImageDB() {
    _fh.close();
  }
  inline int32_t get_nframe()  { return _frame_count; }
  inline int32_t get_nchan()   { return _fileh.nchan; }
  inline int32_t get_nstokes() { return _nstokes;     }
  inline int32_t get_ngrid()   { return _fileh.ngrid; }
  inline void seek(uint32_t idx) {
    _fh.seekg(sizeof(_fileh) + idx*_frame_size, std::ios::beg);
    if( _fh.good() ) {
      _frame_idx = idx;
    }
  }
  OrvilleImage* read();
};

#endif // __INCLUDE_OIMS_HPP
