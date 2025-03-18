from .OrvilleImageHDF5 import OrvilleImageHDF5
from .legacy.OrvilleImageDB import OrvilleImageDB

class OrvilleImageReader:
    """
    Wrapper around OrvilleImageHDF5 and OrvilleImageDB that allows for
    automatic format selection.
    """
    
    @staticmethod
    def open(filename):
        """
        Factory method to create the appropriate image database handler based
        on the file extension.
        """
        
        if filename.endswith('.oims'):
            return OrvilleImageDB(filename, 'r')
        else:
            return OrvilleImageHDF5(filename, 'r')
