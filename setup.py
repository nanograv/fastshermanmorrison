import os
import sys
import numpy

from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize


requirements = [
    "numpy>=1.16.3",
    "scipy>=1.2.0",
]

# Cython extensions
ext_modules=[
    #Extension('fastshermanmorrison.choleskyext_omp',
    #         ['fastshermanmorrison/choleskyext_omp.pyx'],
    #         include_dirs = [numpy.get_include(), 'fastshermanmorrison/'],
    #         extra_link_args=["-liomp5"],
    #         extra_compile_args=["-O2", "-fopenmp", "-fno-wrapv"]),
    #Extension('fastshermanmorrison.choleskyext',
    #         ['fastshermanmorrison/choleskyext.pyx'],
    #         include_dirs = [numpy.get_include(), 'fastshermanmorrison/'],
    #         extra_compile_args=["-O2", "-fno-wrapv"])  # 50% more efficient!
    Extension('fastshermanmorrison.cython_fastshermanmorrison',
             ['fastshermanmorrison/cython_fastshermanmorrison.pyx'],
             include_dirs = [numpy.get_include(), 'fastshermanmorrison/'],
             extra_compile_args=["-O2", "-fno-wrapv"])  # 50% more efficient!
]

setup(
    name="fastshermanmorrison",
    version='2023.04',
    author="Rutger van Haasteren",
    author_email="rutger@vhaasteren.com",
    packages=["fastshermanmorrison"],
    package_dir={"fastshermanmorrison": "fastshermanmorrison"},
    url="http://github.com/vhaasteren/fastshermanmorrison/",
    license="GPLv3",
    description="PTA analysis software",
    long_description=open("README.md").read() + "\n\n"
                    + "Changelog\n"
                    + "---------\n\n"
                    + open("HISTORY.md").read(),
    package_data={"": ["README", "LICENSE", "AUTHORS.md"]},
    include_package_data=True,
    install_requires=["numpy", "scipy"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    ext_modules = cythonize(ext_modules)
)
