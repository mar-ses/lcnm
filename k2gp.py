import sys
import time
import numpy as np
import pandas as pd

from . import gp_models, lc_preparation
from .gp_models import (NonPositiveDefiniteError, LAPACKError,
                        PriorInitialisationError, OversamplingPopulationError)


# TODO: The whole procedure doesn't work as expected post-update.
# Need to split into N_opt and N_det or whatever the other one is.
# Then have the keywords set the two. Also make sure that on
# long_process=True, the proc_kw default is switched to a long version.

# TODO: basically, do not assume that this is working as expected at
# the moment. Needs to be looked at in detail. Performance on both
# long and short cadence is variable and generally atrocious.

# TODO: try and get a CDPP of 350 on the long cadence, see how low I
# can get on the short cadence

# High-level usage functions
# --------------------------

def detrend_lcf_classic(lcf, proc_kw='ideal', ramp=False, kernel_keyword=None,
                        long_process=False, **rfd_kwargs):
    """Performs a full detrending routine on a lightcurve.

    Args:
        lcf (pd.DataFrame): the lightcurve to detrend; must
            be already initialised.
        proc_kw (str): determines number of samples
            to use:
            'fast' is 600 samples per iteration
            'medium' is 1000 samples per iteration
            'full' is the full sample basis
            'ideal' is 800 samples per iteration, plus
            an ocut value of [5, 5, 4, 3]
            Gets overriden by rfd_kwargs
        **rfd_kwargs (dict): arguments to pass to
            `run_full_detrend`:
            n_samples, n_iters, ocut, full_final, plot_all,
            verbose etc...

    Returns:
        lcf, hp, cdpp
    """

    if long_process and proc_kw == "ideal":
        proc_kw = "long"

    # Parse arguments
    rfd_kwargs = parse_process_keywords(proc_kw, rfd_kwargs)

    # Set up object
    k2_kernel = gp_models.ClassicK2Kernel(keyword=kernel_keyword)

    if not long_process:
        k2_detrender = gp_models.LCNoiseModel3D(
            lcf, k2_kernel, additional_model='ramp' if ramp else None)
    else:
        k2_detrender = gp_models.LLCNoiseModel3D(
            lcf, k2_kernel, additional_model='ramp' if ramp else None)

        # Need to override the n_samples in rfd_kwargs
        # pro/

    # TODO
    try:
        k2_detrender.verify_prior(quiet=False, info_str='setup')
    except:
        import pdb; pdb.set_trace()

    return run_full_detrend(k2_detrender, lcf, **rfd_kwargs)


def detrend_lcf_quasiperiodic(lcf, period=None, proc_kw='ideal',
                              ramp=False, kernel_keyword='hf',
                              long_process=False, **rfd_kwargs):
    """Performs a full detrending routine on a lightcurve.

    Args:
        lcf (pd.DataFrame): the lightcurve to detrend; must
            be already initialised
        proc_kw (str): number of samples to use
            'fast' is 600 samples per iteration
            'medium' is 1000 samples per iteration
            'full' is the full sample basis
            Gets overriden by rfd_kwargs
        **rfd_kwargs (dict): arguments to pass to
            `run_full_detrend`:
            n_samples, n_iters, ocut, full_final, plot_all,
            verbose etc...

    Returns:
        lcf, hp, cdpp
    """

    if long_process and proc_kw == "ideal":
        proc_kw = "long"

    # Parse arguments
    rfd_kwargs = parse_process_keywords(proc_kw, rfd_kwargs)

    if period is None:
        period = lc_preparation.detect_lcf_periodicity(lcf)

    # Set up object
    k2_kernel = gp_models.QuasiPeriodicK2Kernel(period, keyword=kernel_keyword)

    if not long_process:
        k2_detrender = gp_models.LCNoiseModel3D(
            lcf, k2_kernel, additional_model='ramp' if ramp else None)
    else:
        k2_detrender = gp_models.LLCNoiseModel3D(
            lcf, k2_kernel, additional_model='ramp' if ramp else None)

    k2_detrender.verify_prior(quiet=False, info_str='setup')

    return run_full_detrend(k2_detrender, lcf, **rfd_kwargs)


# The main detrending routine (work function)
# -------------------------------------------

def run_full_detrend(k2_detrender, lcf=None, n_samples=None, n_iters=4,
                     ocut=5, evolve=False, full_final=False,
                     plot_all=False, verbose=False):
    """Runs a full detrending process.

    May benefit from being turned into an object.
    Especially if splitting the lcgp and k2_detrender
    objects, it could be added as a method of
    k2_detrender.

    Args:
        k2_detrender (): the detrender object, already
            containing a kernel and initialisation.
        lcf (pd.DataFrame, optional): if None will just use
            the lcf already in k2_detrender. Must
            have been initialised fully.
        n_samples (int or tuple): number of samples to use per
            iteration. If tuple, element denotes the ith
            iteration's n_samples.
        n_iters (int): number of iterations of outlier removal,
            i.e sigma clipping, with subsequent optimisation.
        ocut (float or tuple): the ocut to use in each iteration
        evolve (bool or tuple): whether to evolve. If tuple,
            specifies for each iteration.
        full_final (bool): if True, the last iteration uses the
            full basis, default: False. Overrides n_samples.
        plot_all (bool): if True, each stage's detrended
            lightcurve is plotted
        verbose (bool): if True, prints information on
            hyperparameter optimisation and lightcurves to
            standard output.
        estimate_wn (bool): if True, will attempt to estimate
            the white noise term before the beginning of the run.

    Returns:
        lcf, hp, cdpp
    """

    # Parse arguments
    if np.isscalar(ocut) or ocut is None:
        ocut = (ocut,) * n_iters
    if np.isscalar(n_samples) or n_samples is None:
        n_samples = (n_samples,) * n_iters
    if np.isscalar(evolve):
        evolve = (evolve,) * n_iters
    if full_final:
        n_samples[-1] = None

    if len(ocut) != n_iters or len(n_samples) != n_iters:
        raise ValueError("Number of iterations and n_samples or ocut ",
                         "doesn't match")

    # Peform first initialisation
    if lcf is not None:
        k2_detrender.set_lcf(lcf)

    # Loop over iterations of outlier detection and double optimisation
    for i in range(n_iters):
        if ocut[i] is not None:
            k2_detrender.set_ocut(ocut[i])

        k2_detrender.select_basis(N=n_samples[i], save=True)
        ts = time.time()

        if evolve[i]:
            hpe = k2_detrender.evolve_hp()
        else:
            hpe = k2_detrender.get_hp()

        k2_detrender.verify_prior(info_str="iteration {}".format(i))
        tm = time.time()
        hpf = k2_detrender.optimise_hp_powell(initial_hp=hpe, save=True)
        te = time.time()

        if verbose:
            k2_detrender.detrend_lightcurve(hp=hpf)
            print('~'*65 + '\n',
                  "Iteration {}\n".format(i+1),
                  "Time taken for evolution: {:.1f}\n".format(tm - ts),
                  "Time taken for fmin_powell: {:.1f}\n".format(te - tm),
                  "evolution hp: {}\n".format(hpe),
                  "fmin_powell hp: {}\n".format(hpf),
                  "CDPP: {:.2f}".format(k2_detrender.calc_cdpp()),
                  '\n' + '~'*65 + '\n')

        if plot_all:
            lcf = k2_detrender.detrend_lightcurve(hp=hpf)
            k2_detrender.plot_model(lcf,
                                    show=True,
                                    title="Run {}".format(i+1))

        sys.stdout.flush()

    # The final result
    hp_result = hpf.copy()
    hp_dict = k2_detrender.get_parameter_dict(include_frozen=True)
    lcf = k2_detrender.detrend_lightcurve(hp=hp_result)
    cdpp = k2_detrender.calc_cdpp(ts=lcf, columns='f_detrended')

    return lcf, hp_dict, cdpp


# Process keyword definitions
# ---------------------------

def parse_process_keywords(proc_kw, rfd_kwargs=None):
    """Parses the process key_words into the rfd dictionary.

    Args:
        proc_kw (str): options include:
            - fast, f, quick, short, None
            - medium, m
            - slow, full, f, s
            - i, ideal, transit_search, ts, main
            - ie, ideal_e, main_e, semi_evolve
        rfd_kwargs (dict, optional):
    """

    rfd_kwargs = rfd_kwargs if rfd_kwargs is not None else dict()

    if proc_kw in ('fast', 'f', 'quick', 'short', None):
        if 'n_samples' not in rfd_kwargs:
            rfd_kwargs['n_samples'] = 600
    elif proc_kw in ('medium', 'm'):
        if 'n_samples' not in rfd_kwargs:
            rfd_kwargs['n_samples'] = 1000
    elif proc_kw in ('slow', 'full', 'f', 's'):
        if 'n_samples' not in rfd_kwargs:
            rfd_kwargs['n_samples'] = None
    elif proc_kw in ('i', 'ideal', 'transit_search', 'ts', 'main'):
        if 'n_samples' not in rfd_kwargs:
            rfd_kwargs['n_samples'] = 1000
        if 'ocut' not in rfd_kwargs:
            rfd_kwargs['ocut'] = (5, 5, 4, 4, 4)
            rfd_kwargs['n_iters'] = len(rfd_kwargs['ocut'])
    elif proc_kw in ('ie', 'ideal_e', 'main_e', 'semi_evolve'):
        if 'n_samples' not in rfd_kwargs:
            rfd_kwargs['n_samples'] = 1000
        if 'ocut' not in rfd_kwargs:
            rfd_kwargs['ocut'] = (5, 5, 4, 4, 4)
        if 'evolve' not in rfd_kwargs:
            rfd_kwargs['evolve'] = (True, True, True, False, False)
        if 'full_final' not in rfd_kwargs:
            rfd_kwargs['full_final'] = True
        if 'n_iters' not in rfd_kwargs:
            rfd_kwargs['n_iters'] = len(rfd_kwargs['ocut'])
    elif proc_kw in ('long'):
        if 'n_samples' not in rfd_kwargs:
            rfd_kwargs['n_samples'] = None
        if 'ocut' not in rfd_kwargs:
            rfd_kwargs['ocut'] = (5, 5, 4, 4, 4)
        if 'full_final' not in rfd_kwargs:
            rfd_kwargs['full_final'] = False
        if 'n_iters' not in rfd_kwargs:
            rfd_kwargs['n_iters'] = len(rfd_kwargs['ocut'])
    else:
        raise ValueError("proc_kw not recognised: {}".format(proc_kw))

    if 'n_iters' not in rfd_kwargs:
        rfd_kwargs['n_iters'] = 4

    return rfd_kwargs


# Testing functions
# -----------------

def test_standard():
    """Tests the ideal process on trappist, with the full procedure.

    Transits are not flagged automatically."""

    from global_variables import DATA_DIR

    # Initialise the lightcurve
    lcf = pd.read_pickle(
        '{}/trappist/k2gp200164267-c12-lcf-full.pickle'.format(DATA_DIR))
    lcf['x'] = lcf.x_pos
    lcf['y'] = lcf.y_pos
    lcf = lc_preparation.initialise_lcf(lcf, f_col='f_tpf_pos')
    #lcf = lc_preparation.clean_lcf(lcf)
    #lcf = lc_preparation.remove_outliers_initial(lcf, 0.03)
    # if flag_transits:
    #     _, lcf, _ = lc_preparation.flag_transits(lcf)

    p_flag, period = lc_preparation.detect_lcf_periodicity(lcf)

    if p_flag:
        print("Periodicity detected: {:.02f}d".format(period))
    else:
        print("Periodicity not detected.")

    if not p_flag:
        return detrend_lcf_classic(
            lcf, 'ideal', verbose=True, plot_all=True, evolve=False)
    else:
        return detrend_lcf_quasiperiodic(
            lcf, period, 'ideal', verbose=True, plot_all=True, evolve=False)

def test_periodic():
    """Tests the ideal quasiperiodic process on trappist.

    Transits are not flagged automatically."""

    from global_variables import DATA_DIR

    # Initialise the lightcurve
    lcf = pd.read_pickle(
        '{}/trappist/k2gp200164267-c12-lcf-full.pickle'.format(DATA_DIR))
    lcf['x'] = lcf.x_pos
    lcf['y'] = lcf.y_pos
    lcf = lc_preparation.initialise_lcf(lcf, f_col='f_tpf_pos')
    #lcf = lc_preparation.clean_lcf(lcf)
    # lcf = lc_preparation.remove_outliers_initial(lcf, 0.03)

    # if flag_transits:
    #     _, lcf, _ = lc_preparation.flag_transits(lcf)

    output = lc_preparation.find_periodicity_peak(
        lcf.t, lcf.f_temporal + lcf.f_detrended, plot=True)

    print(output)

    return detrend_lcf_quasiperiodic(
        lcf, period=3, proc_kw='ideal', verbose=True, plot_all=True,
        evolve=False, n_samples=1400)


# Determine numpy printing options
# --------------------------------

# Arrays of floats
float_formatter = lambda x: "{:.4g}".format(x)
np.set_printoptions(formatter={'float_kind':float_formatter})
