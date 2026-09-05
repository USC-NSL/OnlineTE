import scipy.sparse as sp


CPUCOOArray = sp.coo_array
"""Alias for `scipy.sparse.coo_matrix`"""
CPUCSRArray = sp.csr_array
"""Alias for `scipy.sparse.csr_matrix`"""
CPUCSCArray = sp.csc_array
"""Alias for `scipy.sparse.csc_matrix`"""


__all__ = ['CPUCSCArray', 'CPUCOOArray', 'CPUCSRArray']