from setuptools import setup, find_packages

setup(name = 'robot2d',
	  version = '0.0.1',
	  install_requires = ['numpy','matplotlib'],
      packages = find_packages(include = ['robot2d', 'robot2d.*'])
)