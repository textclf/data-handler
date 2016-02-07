from setuptools import setup
from setuptools import find_packages

setup(name='NLPDataHandlers',
      version='0.0.1',
      description='Library for loading datasets for deep learning.',
      author='Luke de Oliveira, Alfredo Lainez',
      author_email='lukedeo@stanford.edu, alainez@stanford.edu',
      url='https://github.com/textclf/data-handler',
      # install_requires=['pandas'],
      packages=find_packages())