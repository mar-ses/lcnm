"""Lightcurve noise modelling (with Gaussian processes)."""

import os
import sys

if not os.environ['HOME'] + '/astro_packages' in sys.path:
    sys.path.append(os.environ['HOME'] + '/astro_packages')

import global_variables
