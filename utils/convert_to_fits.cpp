#include <iostream>
#include <utility>
#include <ctime>
#include <fitsio.h>

#include "oims.hpp"

inline time_t mjd_to_time(double mjd) {
  return (time_t) (mjd - 40587) * 86400;
}

inline std::string mjd_to_string(double mjd) {
  time_t start_t = mjd_to_time(mjd);
  struct tm *ptm = gmtime(&start_t);
  char buffer[27];
  std::snprintf(buffer, 27, "%i-%02i-%02iT%02i:%02i:%02i", ptm->tm_year + 1900, ptm->tm_mon+1, ptm->tm_mday,
                                                           ptm->tm_hour, ptm->tm_min, ptm->tm_sec);
  std::string output = std::string(buffer);
  return output;
}

int main(int argc, char** argv) {
  if( argc < 2 ) {
    std::cout << "Must supply a filename to read from" << std::endl;
    return 1;
  } else {
    std::cout << "Reading from '" << argv[1] << "'" << std::endl;
  }
  
  std::string filename = std::string(argv[1]);
  OrvilleImageDB *buffer = new OrvilleImageDB(filename);
  
  fitsfile *fptr;
  char card[24];
  int status = 0, nkeys, ii;
  double value;
  fits_create_file(&fptr, std::string("test.fits").c_str(), &status);
  
  long naxis = 4;
  long naxes[4] = { buffer->get_ngrid(), buffer->get_ngrid(), buffer->get_nstokes(), buffer->get_nchan() };
  long fpixel = 1;
  long nelements = naxes[0]*naxes[1]*naxes[2]*naxes[3];
  
  OrvilleImage *image = buffer->read();
  while( image != NULL ) {
    std::cout << "Read with start time: " << image->get_start_time() << std::endl;
    
    fits_create_img(fptr, FLOAT_IMG, naxis, naxes, &status);
    
    std::string station = image->get_station();
    ::strncpy(&card[0], station.c_str(), 16);
    fits_update_key(fptr, TSTRING, "TELESCOP", &card[0], "", &status);
    
    value = image->get_start_time();
    std::string dateobs = mjd_to_string(value);
    ::strncpy(&card[0], dateobs.c_str(), 24);
    fits_update_key(fptr, TSTRING, "DATE-OBS", &card[0], "", &status);
    
    value = image->get_int_len() * 86400;
    fits_update_key(fptr, TDOUBLE, "INTTIM", &value, "", &status);
    ::strncpy(&card[0], std::string("SECONDS").c_str(), 16);
    fits_update_key(fptr, TSTRING, "INTTIMU", &card[0], "", &status);
    
    value = 2000.0;
    fits_update_key(fptr, TDOUBLE, "EQUINOX", &value, "", &status);
    
    ::strncpy(&card[0], std::string("RA---SIN").c_str(), 16);
    fits_update_key(fptr, TSTRING, "CTYPE1", &card[0], "", &status);
    fpixel = naxes[0]/2 + 1;
    fits_update_key(fptr, TLONG, "CRPIX1", &fpixel, "", &status);
    std::pair<double,double> center = image->get_center_equ();
    value = center.first;
    fits_update_key(fptr, TDOUBLE, "CRVAL1", &value, "", &status);
    value = image->get_pixel_size() * -1;
    fits_update_key(fptr, TDOUBLE, "CDELT1", &value, "", &status);
    ::strncpy(&card[0], std::string("deg").c_str(), 16);
    fits_update_key(fptr, TSTRING, "CUNIT1", &card[0], "", &status);
    
    ::strncpy(&card[0], std::string("DEC--SIN").c_str(), 16);
    fits_update_key(fptr, TSTRING, "CTYPE2", &card[0], "", &status);
    fpixel = naxes[1]/2 + 1;
    fits_update_key(fptr, TLONG, "CRPIX2", &fpixel, "", &status);
    value = center.second;
    fits_update_key(fptr, TDOUBLE, "CRVAL2", &value, "", &status);
    value = image->get_pixel_size();
    fits_update_key(fptr, TDOUBLE, "CDELT2", &value, "", &status);
    ::strncpy(&card[0], std::string("deg").c_str(), 16);
    fits_update_key(fptr, TSTRING, "CUNIT2", &card[0], "", &status);
    
    ::strncpy(&card[0], std::string("STOKES").c_str(), 16);
    fits_update_key(fptr, TSTRING, "CTYPE3", &card[0], "", &status);
    fpixel = 1;
    fits_update_key(fptr, TLONG, "CRPIX3", &fpixel, "", &status);
    fpixel = 1;
    fits_update_key(fptr, TLONG, "CRVAL3", &fpixel, "", &status);
    fpixel = 1;
    fits_update_key(fptr, TLONG, "CDELT3", &fpixel, "", &status);
    
    ::strncpy(&card[0], std::string("FREQ").c_str(), 16);
    fits_update_key(fptr, TSTRING, "CTYPE4", &card[0], "", &status);
    fpixel = 1;
    fits_update_key(fptr, TLONG, "CRPIX4", &fpixel, "", &status);
    value = image->get_start_freq();
    fits_update_key(fptr, TDOUBLE, "CRVAL4", &value, "", &status);
    value = image->get_bandwidth();// / naxes[3];
    fits_update_key(fptr, TDOUBLE, "CDELT4", &value, "", &status);
    ::strncpy(&card[0], std::string("Hz").c_str(), 16);
    fits_update_key(fptr, TSTRING, "CUNIT4", &card[0], "", &status);
    ::strncpy(&card[0], std::string("TOPOCENT").c_str(), 16);
    fits_update_key(fptr, TSTRING, "SPECSYS", &card[0], "", &status);
    
    ::strncpy(&card[0], std::string("UNCALIB").c_str(), 16);
    fits_update_key(fptr, TSTRING, "BUNIT", &card[0], "", &status);
    value = 1.0;
    fits_update_key(fptr, TDOUBLE, "BSCALE", &value, "", &status);
    value = 0.0;
    fits_update_key(fptr, TDOUBLE, "BZERO", &value, "", &status);
    
    fpixel = 1;
    fits_write_img(fptr, TFLOAT, fpixel, nelements, image->get_image(), &status);
    
    image = buffer->read();
  }
  
  fits_close_file(fptr, &status);
  
  return 0;
}
