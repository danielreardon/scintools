#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 21:44:08 2019

@author: dreardon
"""

from setuptools import setup

setup(
    name='scintools',
    version='0.2',
    description='Tools for analysing pulsar scintillation.',
    long_description="See: `github.com/danielreardon/scintools \
                      <https://github.com/danielreardon/scintools>`_.",
    classifiers=[
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='dynamic-spectrum, pulsar-timing, scintillation,\
              pulsar-scintillation',
    url='https://github.com/danielreardon/scintools',
    author='Daniel J. Reardon',
    author_email='dreardon@swin.edu.au',
    license='MIT',
    packages=['scintools'],
    install_requires=['numpy', 'scipy', 'matplotlib', 'corner', 'lmfit', 'astropy', 'emcee', 'bilby == 1.2.0', 'scikit-image == 0.16.2'],
    include_package_data=True,
    zip_safe=False,
)
