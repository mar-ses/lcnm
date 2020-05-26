"""Preparation of lightcurves for detrending; not for analysis.

Tasks: extraction of lcfs, raw outlier detection and removal,
initialisation of lcfs, transit flagging.

Sections:
---------
1. Preparation of lcf files (initialisation, outlier detection...).
2. Extraction of lcf files.
3. White noise estimation.
4. Old routines.

Also, for lcf subselection.

TODO: separate sc and lc; for sc, just have sc_ as a prefix.

Current version (08/05/18): moving completely to full-lcf treatment.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from astropy.io import fits
from astropy.timeseries import LombScargle

from . import gp_models, lc_utils
# from .gp_utils import ravel_trappist_times

# Will only work if lcnm package is imported
from global_variables import HOME_DIR, UB_MANUAL








def initialise_lcf(lcf, f_col='f', with_xy=True, pos_suffix=None):
    """Initialises columns, normalises flux, removes null values.

    Generally required before detrending. Must not have been
    normalised to 0.0 before. Puts the flux in column 'f'
    whatever the previous case. Expects positions in 'x' and 'y'.
    Also initialises the o_flag and t_flag columns.

    Args:
        lcf (pd.DataFrame):
        f_col (str): column to use for flux, default: 'f'

    Returns:
        lcf (pd.DataFrame)
    """

    if pos_suffix is not None:
        lcf['x'] = lcf['x' + pos_suffix]
        lcf['y'] = lcf['y' + pos_suffix]

    if with_xy:
        lcf = lcf.loc[~lcf[f_col].isnull() \
                    & ~lcf['x'].isnull() \
                    & ~lcf['y'].isnull() \
                    & ~lcf['t'].isnull()]
    else:
        lcf = lcf.loc[~lcf[f_col].isnull() \
                    & ~lcf['t'].isnull()]

    lcf = lcf.assign(o_flag=False, f_abs=lcf[f_col].copy())

    if 't_flag' not in lcf.columns:
        lcf = lcf.assign(t_flag=False)

    if lcf.t.duplicated().any():
        raise ValueError("Duplicated exposures in lightcurve.")

    lcf = lcf.assign(f=lcf[f_col] / np.nanmedian(lcf[f_col]))

    # Store the raw values for later
    lcf = lcf.assign(f_raw=lcf.f.copy())
    lcf = lcf.sort_values(by='t')
    lcf.index = range(len(lcf.index))

    return lcf




# Manually produced full_lcf files (lcf-full.pickle)
# --------------------------------------------------

def read_full_lcf(filepath, flux_source='pdc'):
    """Reads a full lcf pickle file and returns the lightcurve.

    pos:	f_tpf_pos with x_pos
    track:	f_tpf_track with x_track
    static:	f_tpf_static with x_pos
    pdc:	f_pdc with x_pos
    sap:	f_sap with x_pos

    Arguments:
        filepath (str): the full path to the lcf
        flux_source (str), 'pos': which photometric configuration to initialise
            to the default values. Options: pos, track, static, pdc, sap. None
            doesn't initialise any source to the default values.

    Returns:
        lcf (pd.DataFrame): with the 'x', 'y' and 'f' columns set to the source values. Nulls are currently not removed.

    Raises:
        FileNotFoundError
        NullLightCurveError
    """

    lcf = pd.read_pickle(filepath)

    if flux_source in ('static', 'stat', 'stationary', 's'):
        lcf['f'] = lcf.f_tpf_static
        lcf['x'] = lcf.x_pos
        lcf['y'] = lcf.y_pos
    elif flux_source in ('track', 't', 'manual'):
        lcf['f'] = lcf.f_tpf_track
        lcf['x'] = lcf.x_pos
        lcf['y'] = lcf.y_pos
    elif flux_source in ('pdc',):
        lcf['f'] = lcf.f_pdc
        lcf['x'] = lcf.x_pos
        lcf['y'] = lcf.y_pos
    elif flux_source in ('sap',):
        lcf['f'] = lcf.f_sap
        lcf['x'] = lcf.x_pos
        lcf['y'] = lcf.y_pos
    else:
        lcf['f'] = lcf.f_tpf_pos
        lcf['x'] = lcf.x_pos
        lcf['y'] = lcf.y_pos

    if lcf.f.isnull().all():
        raise NullLightCurveError(
            "All-NaN values for column '{}' in the lightcurve of: {}"
            "".format(flux_source, filepath))

    # Attempts to write-in the source.
    lcf.flux_source = flux_source

    return lcf

def find_full_lcf(epic, campaign=None, flux_source='pos', dir_loc=UB_MANUAL):
    """Finds the full lcf pickle file and returns the lightcurve.

    Arguments:
        epic (int or str):
        campaign (int or str, optional):
        flux_source (str), 'pos': which photometric configuration to initialise
            to the default values. Options: pos, track, static, pdc, sap. None
            doesn't initialise any source to the default values.
        dir_loc (str): where to search for the epic number.

    Returns:
        lcf (pd.DataFrame): with the 'x', 'y' and 'f' columns set to
            the source values. Nulls are currently not removed.

    Raises:
        FileNotFoundError
        NullLightCurveError
    """

    if not dir_loc.startswith('/'):
        dir_loc = "{}/{}".format(HOME_DIR, dir_loc)

    cstr = str(int(epic))

    if campaign is not None:
        cstr = cstr + '-c' + str(int(campaign))

    for directory, _, filename in os.walk(dir_loc):
        for f in filename:
            if cstr in f and f.endswith('lcf-full.pickle'):
                try:
                    return read_full_lcf("{}/{}".format(directory, f), flux_source)
                except NullLightCurveError:
                    raise NullLightCurveError("All-NaN values in column '{}' of EPIC: {}".format(flux_source, epic))

    raise FileNotFoundError('{} not found.'.format(cstr))


# In-depth pre-detrending analysis
# --------------------------------

def quick_detrend(lcf, long_timescale=False, long_process=False):
    """Performs a very fast initial detrending to see time variation.

    Args:
        lcf (pd.DataFrame): must be initialised

    Returns:
        lcf (pd.DataFrame): detrended
    """

    lcf = lcf.copy()

    if long_timescale:
        kernel_keyword = 'long'
    else:
        kernel_keyword = None

    k2_kernel = gp_models.ClassicK2Kernel(keyword=kernel_keyword)

    if not long_process:
        detrender = gp_models.LCNoiseModel3D(lcf, k2_kernel)
    else:
        detrender = gp_models.LLCNoiseModel3D(lcf, k2_kernel)

    detrender.verify_prior(quiet=False)
    detrender.select_basis(1000)
    detrender.optimise_hp_powell(save=True)
    detrender.select_basis()

    lcf = detrender.detrend_lightcurve()
    lcf['o_flag'] = detrender.mask_outliers(ocut=4)

    return lcf

def detect_ramp(lcf, period=None, diagnose=False, long_process=False):
    """Determines if there is a ramp in the beginning of the data.


    This may be useless though, as it's just an outlier;
    perhaps this should be done after detrending anyway.

    Actually, it is likely best to just remove it from f_detrended
    by checking outlier numbers. In any case, o_flag must be
    considered somehow.

    Args:
        lcf
        period (float): if given, will fit quasiperiodic
        diagnose (bool): plots fit if True

    Returns:
        ramp_flag
    """

    # Remove initial points
    lcf = lcf.copy()

    if period is None or np.isnan(period):
        k2_kernel = gp_models.ClassicK2Kernel()
    else:
        keyword = None if period > 1.0 else 'hf'
        k2_kernel = gp_models.QuasiPeriodicK2Kernel(period, keyword=keyword)

    if not long_process:
        detrender = gp_models.LCNoiseModel3D(
            lcf, k2_kernel, ocut=5, additional_model='ramp')
    else:
        detrender = gp_models.LLCNoiseModel3D(
            lcf, k2_kernel, ocut=5, additional_model='ramp')


    detrender.verify_prior(quiet=False)
    detrender.select_basis(1000)
    detrender.optimise_hp_powell(save=True)
    detrender.select_basis(1000)
    detrender.optimise_hp_powell(save=True)

    lcf_full = detrender.detrend_lightcurve()
    A = detrender.get_parameter('A')

    temporal_range = max(lcf_full.f_temporal) - min(lcf_full.f_temporal)

    ramp_flag = A > temporal_range/2

    if diagnose:
        print("Ramp flagged = {}".format(ramp_flag))
        std_range = np.percentile(lcf_full.f, [16, 84])
        ramp_flag_2 = A > 0.5*(std_range[1] - std_range[0])
        print("Ramp flagged 2 = {}".format(ramp_flag_2))
        import pdb; pdb.set_trace()
        o_flag = detrender.mask_outliers(ocut=4)
        lcf_full['o_flag'] = o_flag
        detrender.plot_model(show=True)

    return ramp_flag

def detect_lcf_periodicity(lcf, *args,  long_process=False, **kwargs):
    """Flags and estimates periodicity in stellar variability.

    First detrends the lightcurve, and then computes the Lomb-Scargle.
    This is a computationally significant function.

    Args:
        lcf
        *args, **kwargs: to analysis.find_periodicity_peak
            plims=None, plot=False

    Returns:
        [threshhold_flag, period]
    """

    lcf_d = quick_detrend(lcf, long_timescale=True, long_process=long_process)

    # Transits can actually confuse the period detection
    # This step attempts to remove them
    lcf_d = lcf_d[~lcf_d.o_flag]

    # f_tvar = lcf_d.f_temporal + lcf_d.f_detrended \
    # 	   - np.nanmedian(lcf_d.f_detrended)
    # The aim is for the detrender to remove long trends
    # and expose the short period variability
    f_tvar = lcf_d.f_detrended

    period, flag, _, _ = find_periodicity_peak(lcf_d.t, f_tvar,
                                               *args, **kwargs)

    return flag, period

def find_periodicity_peak(t, f, plims=None, plot=False):
    """Calculates the Lomb-Scargle periodogram of a timeseries.

    Default threshold for the false-alarm probability is now
    set to 10**-30 (k2sc is 1e-50). Perhaps a more
    specific metric should be used (i.e amplitude of wave
    vs white noise).

    Args:
        t
        f
        plims (tuple): upper and lower bound for periods
            to calculate the periodogram on. Default: [0.5, 15]

    Returns:
        max_period (float): value of the most valid peak
        threshhold_flag (bool): True if passes the threshold
        pfa (float): probability of false alarm
            P(peak max | gaussian noise)
        lsp (pd.DataFrame): 'frequency', 'period', 'power'
    """

    threshold_fa = 1e-30

    plims = plims if plims is not None else (0.5, 15)

    ls = LombScargle(t, f)

    freqs, power = ls.autopower(minimum_frequency=1/max(plims),
                                maximum_frequency=1/min(plims))
    periods = 1.0 / freqs
    lsp = pd.DataFrame({'frequency':freqs,
                        'period':periods,
                        'power':power})

    max_period = periods[np.argmax(power)]

    pfa = ls.false_alarm_probability(power.max())

    threshhold_flag = True if pfa < threshold_fa else False

    if plot:
        fig, ax = plt.subplots(3)

        ax[0].plot(t, f, 'k.')
        ax[0].set_xlabel('t')
        ax[0].set_ylabel('f')

        ax[1].plot(lc_utils.fold_on_first(t, max_period), f, 'k.')
        ax[1].set_xlabel('t')
        ax[1].set_ylabel('f')

        ax[2].plot(periods, power, 'k-')
        ax[2].set_xlabel('period')
        ax[2].set_ylabel('power')
        fig.show()

    return max_period, threshhold_flag, pfa, lsp
    

# White noise estimation and initial hyperparameters
# --------------------------------------------------

def estimate_sigma(f, n_chunks=10000, chunk_size=8):
    """Finds the true standard deviation of the light curve.

    Attempts to remove the effects of the systematic trends
    and outliers, using a sigmaclipping procedure, and taking
    a lower percentile of the resulting sigmas.

    Args:
        f (pd.Series): the flux column
        n_chunks (int, 100000): number of subsets to use
        chunk_size (int, 8): number of points per subset

    Returns:
        sigma (float): the standard deviation of the lightcurve
    """

    # Split the data into chunks.
    if isinstance(f, (pd.Series, pd.DataFrame)):
        start_idxs = f[:-chunk_size].sample(n_chunks, replace=True).index
    else:
        start_idxs = np.random.choice(
            np.arange(0, len(f)), n_chunks, replace=True)

    sigma_list = []
    for index in start_idxs:
        chunk = f[index : index+chunk_size]
        sigma_list.append(stats.sigmaclip(chunk)[0].std())
    sigma = np.percentile(sigma_list, 30)

    return sigma

def get_white_noise(lcf, chunk_size=10):
    """Estimates the initial white noise parameter value.

    This follows the george parameterisation of a gp.
    The estimate is slightly reduced for more aggressive
    initial outlier detection, and to account for systematic
    error inclusion.

    Aggressive sigma clipping is used to estimate the variance.
    With an attempt to remove temporal variance.
    """

    # red_factor 3 worked but is generally an underestimate.
    # red_factor 1 is generally the correct amount of white noise.
    # However, it may be prone to underestimating outliers.

    red_factor = 1.5
    return 2*np.log(estimate_sigma(lcf.f, chunk_size=chunk_size)/red_factor)

def mask_flares(f, factor=6.0, n_largest=40):
    """Rough removal of flare-like points.

    Use only in the beginning.

    Arguments:
        f (np.array): the flux values
        factor (float=6.0): the factor of sigma to count as a potential
            flare
        n_largest (int): maximum number of point to remove as potential
            flares

    Returns:
        mask (np.ndarray)
    """

    if isinstance(f, (pd.DataFrame, pd.Series)):
        f = f.values

    n_largest = 40

    rms = estimate_sigma(
        f, n_chunks=5, chunk_size=min(2500, len(f)-3))
    mask = f > (np.nanmedian(f) + 6*rms)

    if np.sum(mask) > n_largest:
        largest_idx = np.argsort(-f)[:n_largest]
        mask = False & mask
        mask[largest_idx] = True

    return mask

def initial_hp(lcf):
    """Provides initial values for the hyperparameters."""

    white_noise = get_white_noise(lcf)
    return [white_noise, -12.86, -3.47, -4.34, -12.28, 2.32]

def estimate_variability(t, f):
    """Estimates the average absolute long-term variability gradient.

    Args:
        t, f (np.arrays)
        timescale (float)
    """

    raise NotImplementedError


# Utilities and exception
# -----------------------

class NullLightCurveError(Exception):
    pass