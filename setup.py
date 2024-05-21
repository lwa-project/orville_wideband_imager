import os
import glob
import shutil

from setuptools import setup, find_namespace_packages

def update_bad_freq():
    setup_script_path = os.path.dirname(os.path.abspath(__file__))
    bad_freq = os.path.join(setup_script_path, 'bad_freq.txt')
    if os.path.exists(bad_freq):
        bad_freq_toolkit_path = os.path.join(setup_script_path, 'src', 'lsl_toolkits', 'OrvilleImage')
        shutil.copy(bad_freq, os.path.join(bad_freq_toolkit_path, '_bad_freq.py'))

update_bad_freq()

setup(name                 = "lsl-toolkits-orvilleimage",
      version              = "0.4.0",
      description          = "LSL Toolkit for Orville Image Database Files",
      long_description     = "LWA Software Library reader for PASI Image Database files",
      author               = "J. Dowell, S. Varghese, and G. B. Taylor",
      author_email         = "jdowell@unm.edu",
      license              = 'BSD3',
      packages             = find_namespace_packages(where='src/', include=['lsl_toolkits.OrvilleImage']),
      package_dir          = {'': 'src'},
      scripts              = glob.glob('scripts/*.py'),
      python_requires      = '>=3.6',
      install_requires     = ['numpy', 'scipy', 'astropy', 'lsl'],
      zip_safe             = False,
      test_suite           = "tests")
