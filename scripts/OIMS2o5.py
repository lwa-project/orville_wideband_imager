#!/usr/bin/env python3

import os
import sys
import numpy as np
import argparse

from lsl.common.progress import ProgressBarPlus

from lsl_toolkits.OrvilleImage import OrvilleImageHDF5
from lsl_toolkits.OrvilleImage.legacy import OrvilleImageDB


def main(args):
    compression = 'gzip' if args.compression else None
    
    for filename in args.filename:
        outname = os.path.basename(filename)
        outname = os.path.splitext(outname)[0]+'.o5'
        outname = os.path.join(args.output_dir, outname)
        
        print(f"Converting {os.path.basename(filename)} to {os.path.basename(outname)}...")
        with OrvilleImageDB(filename, 'r') as db:
            ## Make sure we scrub bytes from the .oims metadata
            imager_version = db.header.imager_version
            if isinstance(imager_version, bytes):
                imager_version = imager_version.decode()
            station = db.header.station
            if isinstance(station, bytes):
                station = station.decode()
                
            ## Setup the progress bar
            pb = ProgressBarPlus(max=db.nint)
            sys.stdout.write(pb.show()+'\r')
            sys.stdout.flush()
            
            ## Convert!
            with OrvilleImageHDF5(outname, mode='w',
                                  imager_version=imager_version, station=station,
                                  compression=compression,
                                  compression_opts=args.compression_opts) as o5:
                for (metadata,data) in db:
                    ### Make sure we scrub bytes from the .oims metadata
                    for key in metadata:
                        if isinstance(metadata[key], bytes):
                            metadata[key] = metadata[key].decode()
                           
                    if isinstance(data, np.ma.core.MaskedArray):
                        mask = data.mask[:,0,0,0,0]
                        data = data.data
                    else:
                        mask = None 
                    o5.add_image(metadata, data, mask=mask)
                    
                    pb.inc(amount=1)
                    sys.stdout.write(pb.show()+'\r')
                    sys.stdout.flush()
                    
            ## Close out the progress bar
            sys.stdout.write(pb.show()+'\r')
            sys.stdout.write('\n')
            sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='convert one or more .oims files into the .o5 format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str, nargs='+',
                        help='filename to convert')
    parser.add_argument('-o', '--output-dir', type=str, default='.',
                       help='directory to write .o5 files to')
    parser.add_argument('-c', '--compression', action='store_true',
                        help='enable gzip compression on the .o5 file')
    parser.add_argument('-l', '--compression-level', type=int,
                        help='compression level is compression is to be used')
    args = parser.parse_args()
    main(args)
