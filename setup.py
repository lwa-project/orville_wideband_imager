import glob

from setuptools import setup, Extension, Distribution, find_packages


setup(name="OrvilleImager",
      version="1.0",
      description="""The Orville Wideband Imager is a realtime GPU-based all-sky imager for the output of the Advanced Digitial Processor (ADP) broadband correlator that runs at LWA-SV. Orville receives visibility data from ADP for 32,896 baselines, images the data, and writes the images to the disk in a binary frame-based format called "OIMS". The imaging is performed using a w-stacking algorithm for the non-coplanarity of the array. For each image, the sky is projected onto the two dimensional plane using orthographic sine projection. To reduce the number of w-planes needed during w-stacking, the phase center is set to a location approximately 2 degrees off zenith that minimizes the spread in the w coordinate. The gridding operation is based on the Romein gridder implemented as part of the EPIC project. Every 5 seconds, the imager produces 4 Stokes (I, Q, U and V) images in 198 channels, each with 100 kHz bandwidth.""",
      author="J. Dowell, S. Varghese, and G. B. Taylor",
      author_email="jdowwell@unm.edu",
      packages=["OrvilleImager"],
      scripts = glob.glob('scripts/*.py'),
      python_requires ='>3.6',
      install_requires = ['bifrost','lsl'],
#       install_requires = ['numpy','matplotlib','lsl','scipy','astropy','datetime','bifrost','ephem','tqdm','lsl-toolkits-pasiimage','paramiko'],
      test_suite="tests")


