# lcnm

## Introduction

Package for modelling and removing correlated noise in lightcurves, specifically pointing drift systematics, and stellar variability. Uses a Gaussian process regression to model the noise, hyperparameters aimed specifically for detrending K2 and TESS lightcurves. Intended for noise removal prior to performing a transit search.

## Technologies

python 3, uses the george package (https://george.readthedocs.io/en/latest/)

## Usage

Interface is not developed yet. `/lcnm/k2gp.py` contains the main functions that perform the detrending automatically. For a lightcurve in the form a `pandas.DataFrame`, with the columns 't', 'x', 'y', 'f' referring to the time, x-position, y-position and total flux/brightness of a star respectively:

```python
from lcnm import k2gp
from lcnm import lc_preparation

lcf = lc_preparation.initialise_lcf(lcf, f_col='f')

lcf_detrended = k2gp.detrend_lcf_classic(lcf)
```

`lcf_detrended` will contain the same columns as `lcf`, plus: `'f_temporal'`, `'f_spatial'`, `'f_detrended'`, `'o_flag'`. `'f_detrended'` contains the "flattened" lightcurve, minus time-correlated noise (long timescales) and x,y-correlated noise.

This hasn't been tested on other computers yet.
