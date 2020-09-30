import numpy as np
import pandas as pd

import cdpp


def calc_cdpp(lcf, verbose=False, column='f_detrended', raw=True,
              cadence="lc"):
    '''
    Calculate the cdpp for the points which are NOT flagged as transits or outliers.
    '''

    if np.median(lcf[column]) < 0.1:
        offset = 1.0
    else:
        offset = 0.0

    if raw:
        cdpp_value = cdpp.CDPP(lcf[column].values + offset)
    else:
        lcf = lcf[~lcf.t_flag & ~lcf.o_flag]
        cdpp_value = cdpp.CDPP(lcf[column].values + offset, cadence=cadence)

    if verbose:
        print('\n###--------------------------------------')
        print('\tCDPP for lightcurve, {} = {}'.format(column, cdpp_value))
        print('###--------------------------------------\n')

    return cdpp_value


# Utilities
# ---------

def fold_on_first(t, period):
    """Folds on the first value.
    """

    tf = np.empty(len(t), dtype=float)
    tf[:] = np.nan

    t0 = min(t)
    t_length = max(t) - min(t)
    num_folds = int(t_length // period) + 1

    for i in range(num_folds + 1):
        mask = ((t - t0) < (i+1)*period) & ((t - t0) >= i*period)
        if isinstance(mask, pd.Series):
            mask = mask.values
        tf[mask] = t[mask] - i*period

    return tf

def bin_int(f, nbin):
    pass