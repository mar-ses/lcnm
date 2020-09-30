"""Main tools for plotting lightcurves and showing noise models."""

import numpy as np
import matplotlib.pyplot as plt

from . import lc_utils

# Main lightcurve plots
# ---------------------

def plot_gp_model(lcf, show=True, title=None, flag_outliers=True,
                  flag_transits=True, t_bjd=True, bin_detrended=False):
    '''Full plot of GP noise model components.'''

    if t_bjd is True and 't_bjd' in lcf.columns:
        tcol = 't_bjd'
    else:
        tcol = 't'

    f_0 = np.nanmedian(lcf.f_detrended)

    if 'f_model' not in lcf.columns:
        lcf['f_model'] = f_0

    if 'o_flag' in lcf.columns and 't_flag' in lcf.columns and flag_transits and flag_outliers:
        lcf_clean = lcf[~lcf.o_flag & ~lcf.t_flag]
    elif 'o_flag' in lcf.columns and flag_outliers:
        lcf_clean = lcf[~lcf.o_flag]
    elif 't_flag' in lcf.columns and flag_transits:
        lcf_clean = lcf[~lcf.t_flag]
    else:
        lcf_clean = lcf

    # Choose alpha and markersize based on number of points
    if len(lcf) < 6000:
        malpha = 0.7
        msize = 4
    else:
        malpha = 0.4
        msize = 2.5

    # Main plotting routine.
    # ----------------------
    fig, ax = plt.subplots(4, sharex=True)

    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)

    # The pixel position vector
    ax[0].plot(
        lcf[tcol], lcf.x, 'r.', lcf[tcol], lcf.y, 'b.',
        alpha=malpha, markersize=msize)

    # The undetrended light curve data points
    ax[1].plot(
        lcf_clean[tcol], lcf_clean.f, 'k.', alpha=malpha, markersize=msize)
    ax[1].plot(lcf[tcol],
               lcf.f_spatial + lcf.f_temporal + lcf_clean.f_model,
               'r-', alpha=0.7, linewidth=1.2)
    if flag_outliers:
        ax[1].scatter(lcf.loc[lcf.o_flag, tcol],
                      lcf[lcf.o_flag].f,
                      facecolors='r',
                      edgecolors='face',
                      marker='.',
                      alpha=malpha-0.15,
                      s=msize**2)
    if flag_transits:
        ax[1].scatter(lcf.loc[lcf.t_flag, tcol],
                      lcf[lcf.t_flag].f,
                      facecolors='b',
                      edgecolors='face',
                      marker='.',
                      alpha=malpha+0.1,
                      s=msize**2)

    # The lightcurve, detrended from the spatial component
    ax[2].plot(lcf_clean[tcol],
               lcf_clean.f - lcf_clean.f_spatial - lcf_clean.f_model + f_0,
               'k.', alpha=malpha, markersize=msize)
    ax[2].plot(lcf[tcol], lcf.f_temporal + f_0,
               'r-', alpha=0.7, linewidth=1.2)
    if flag_outliers:
        ax[2].scatter(lcf.loc[lcf.o_flag, tcol],
                      lcf[lcf.o_flag].f - lcf[lcf.o_flag].f_spatial,
                      facecolors='r',
                      edgecolors='face',
                      marker='.',
                      alpha=malpha-0.15,
                      s=msize**2)
    if flag_transits:
        ax[2].scatter(lcf.loc[lcf.t_flag, tcol],
                      lcf[lcf.t_flag].f - lcf[lcf.t_flag].f_spatial,
                      facecolors='b',
                      edgecolors='face',
                      marker='.',
                      alpha=malpha+0.1,
                      s=msize**2)

    # The lightcurve, detrended from all but the white noise
    ax[3].plot(
        lcf_clean[tcol], lcf_clean.f_detrended, 'k.',
        alpha=malpha, markersize=msize)
    if flag_outliers:
        ax[3].scatter(lcf.loc[lcf.o_flag, tcol],
                      lcf[lcf.o_flag].f_detrended,
                      facecolors='r',
                      edgecolors='face',
                      marker='.',
                      alpha=malpha-0.15,
                      s=msize**2)
    if flag_transits:
        ax[3].scatter(lcf.loc[lcf.t_flag, tcol],
                      lcf[lcf.t_flag].f_detrended,
                      facecolors='b',
                      edgecolors='face',
                      marker='.',
                      alpha=malpha+0.1,
                      s=msize**2)
    if bin_detrended:
        pass

    # BEFORE (don't know why I didn't just do lcf.f_detrended)

    # ax[3].plot(lcf_clean[tcol],
    # 		   lcf_clean.f - lcf_clean.f_spatial - lcf_clean.f_temporal,
    # 		   'k.', alpha=0.5)
    # if flag_outliers:
    # 	ax[3].scatter(lcf.loc[lcf.o_flag, tcol],
    # 				  lcf[lcf.o_flag].f - lcf[lcf.o_flag].f_spatial\
    # 				  					- lcf[lcf.o_flag].f_temporal,
    # 				  facecolors='r',
    # 				  edgecolors='face',
    # 				  marker='.',
    # 				  alpha=0.4)
    # if flag_transits:
    # 	ax[3].scatter(lcf.loc[lcf.t_flag, tcol],
    # 				  lcf[lcf.t_flag].f - lcf[lcf.t_flag].f_spatial\
    # 									- lcf[lcf.t_flag].f_temporal,
    # 				  facecolors='b',
    # 				  edgecolors='face',
    # 				  marker='.',
    # 				  alpha=0.8,
    # 				  s=15)

    if not t_bjd or max(lcf[tcol]) < 4000:
        ax[3].set_xlabel('Time, BKJD')
    else:
        ax[3].set_xlabel('Time, BJD')
    ax[3].set_xlim(min(lcf[tcol]), max(lcf[tcol]))

    ax[1].set_ylabel('Normalised raw flux')
    ax[2].set_ylabel('Time-correlated component')
    ax[3].set_ylabel('Fully detrended flux')

    if title is not None:
        fig.suptitle(title) # + '\n' + np.array_str(GP.get_vector()))

    if show:
        plt.show()
    else:
        fig.show()

    return fig, ax

def plot_lcf_single(lcf, clean_lcf=True, show=True,
                    folded_period=None):
    """Plots the three detrendings on a single figure.

    If no detrendings are there, will only plot the raw data.
    """

    if clean_lcf and 'o_flag' in lcf.columns:
        lcf = lcf[~lcf.o_flag]

    if folded_period is None:
        t = lcf.t
    else:
        t = lc_utils.fold_on_first(lcf.t.values, folded_period)

    fig, ax = plt.subplots()

    ax.plot(t, lcf.f,
            color='0.7', marker='.', linestyle='none',
            alpha=0.4, zorder=-5)

    if 'f_temporal' in lcf.columns and 'f_detrended' in lcf.columns:
        ax.plot(t,
                lcf.f_temporal + lcf.f_detrended - np.nanmedian(lcf.f_detrended),
                color='0.3', marker='.', linestyle='none',
                alpha=0.7, zorder=-4)

    if 'f_detrended' in lcf.columns:
        ax.plot(t, lcf.f_detrended,
                color='b', marker='.', linestyle='none',
                alpha=0.7, zorder=0)

    if show:
        plt.show()
    else:
        fig.show()

def plot_lcf(lcf):
    """ """
    raise NotImplementedError
