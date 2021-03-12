from distutils.core import setup
from Cython.Build import cythonize

setup(name='Sasha Wang', ext_modules=cythonize("sw.pyx"))
