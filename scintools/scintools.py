#!/usr/bin/env python

"""
scintools.py
----------------------------------
Launch scintools from command line
"""

import argparse

parser = argparse.ArgumentParser(description='Scintools: Scintillation tools')
parser.add_argument('-f', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
