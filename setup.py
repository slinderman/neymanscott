#!/usr/bin/env python
from distutils.core import setup

setup(name='neymanscott',
      version='0.1',
      description='Bayesian inference for Neyman-Scott processes',
      author='Scott Linderman and Yixin Wang',
      author_email='scott.linderman@stanford.edu',
      url='http://www.github.com/slinderman/pyhawkes',
      install_requires=['numpy', 'scipy', 'matplotlib'],
      packages=['neymanscott']
     )
