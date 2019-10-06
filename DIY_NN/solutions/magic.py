import numpy as np

def Matrix(*a):
    if len(a)==1 and isinstance(a[0], np.ndarray):
        a = a[0]
    return np.array([[float(x) for x in r] for r in a])

def Vector(*a):
    if len(a)==1 and isinstance(a[0], np.ndarray):
        a = a[0]
    return np.array([float(x) for x in a]).reshape(-1,1)

# Black magic
from IPython.display import Latex, SVG, display
from IPython.core.interactiveshell import InteractiveShell

def ndarray_to_latex(arr): 
    if len(arr.shape)==1: 
        arr=arr.reshape(1,-1)
    if len(arr.shape) == 2:
        if max(arr.shape) > 30:
            return None
        str_arr = np.vectorize("{:.3f}".format)(arr)
        return r'\begin{{pmatrix}}{}\end{{pmatrix}}'.format(r'\\ '.join(map('&'.join, str_arr))) 
    if len(arr.shape) == 3 and arr.shape[2]==1:
        if max(arr.shape) > 30:
            return None
        arr = arr[:,:,0]
        str_arr = np.vectorize("{:.3f}".format)(arr)
        return r'\begin{{bmatrix}}[{}]\end{{bmatrix}}'.format(
            r']\\ ['.join(map('&'.join, str_arr))) 
    return None
sh = InteractiveShell.instance()
sh.display_formatter.formatters['text/latex'].type_printers[np.ndarray]=ndarray_to_latex