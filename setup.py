from setuptools import Extension, setup
#as taught in class 

module = Extension("mykmeanssp",['spkmeans.c','spkmeansmodule.c'])  
setup(name='mykmeanssp',
      version='1.0',
      description='Python Macdonald Wrapper for custom C extension',
      ext_modules=[module]) 
#as taught in class 