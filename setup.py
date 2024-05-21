import glob

from setuptools import setup, Extension, Distribution, find_packages

setup(name                 = "lsl-toolkits-orvilleimage",
      version              = "0.2.0",
      description          = "LSL Toolkit for Orville Image Database Files",
      long_description     = "LWA Software Library reader for PASI Image Database files",
      author               = "J. Dowell, S. Varghese, and G. B. Taylor",
      author_email         = "jdowell@unm.edu",
      license              = 'BSD3',
      packages             = find_packages(exclude="tests"),
      namespace_packages   = ['lsl_toolkits',],
      scripts              = glob.glob('scripts/*.py'),
      python_requires      = '>=3.6',
      install_requires     = ['numpy', 'lsl'],
      include_package_data = True,
      zip_safe             = False,
      test_suite           = "tests")
