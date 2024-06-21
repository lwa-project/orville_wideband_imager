import os
import glob
import shutil

from setuptools import setup, find_namespace_packages

setup(name                 = "lsl-toolkits-orvilleimage",
      version              = "0.4.0",
      description          = "LSL Toolkit for Orville Image Database Files",
      long_description     = "LWA Software Library reader for Orville Image Database files",
      author               = "J. Dowell, S. Varghese, and G. B. Taylor",
      author_email         = "jdowell@unm.edu",
      license              = 'BSD3',
      packages             = find_namespace_packages(where='src/', include=['lsl_toolkits.OrvilleImage']),
      package_dir          = {'': 'src'},
      package_data         = {'lsl_toolkits.OrvilleImage': ['data/*.txt']},
      scripts              = glob.glob('scripts/*.py'),
      python_requires      = '>=3.6',
      install_requires     = ['numpy', 'scipy', 'astropy', 'lsl'],
      zip_safe             = False,
      test_suite           = "tests")
