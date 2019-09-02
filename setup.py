from setuptools import setup, find_packages
import sys, os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gym_pomdp_wrappers'))

setup(name='gym_pomdp_wrappers',
      version='1.0.0',
      description='POMDP wrappers for OpenAI Gym',
      author='Stephan Weigand',
      license="MIT",
      packages=[package for package in find_packages()
                if package.startswith('gym_pomdp_wrappers')],
      python_requires='>=3.5',
      install_requires=['gym', 'numpy'])
