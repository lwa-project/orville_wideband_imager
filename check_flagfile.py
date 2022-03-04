#!/usr/bin/env python3

"""
Simple script to read in and check a Orville flagfile.
"""

import os
import sys


def main(args):
    filename = args[0]
    with open(filename, 'r') as fh:
        for line in fh:
            line = line.strip().rstrip()
            if len(line) < 3:
                continue
            if line[0] == '#':
                continue
                
            try:
                f = float(line)*1e6
                print(f"Found a flag for {f/1e6:.3f} MHz")
            except ValueError:
                print(f"WARNING: invalid line - '{line}'")


if __name__ == '__main__':
    main(sys.argv[1:])
