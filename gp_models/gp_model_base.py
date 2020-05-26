"""The base object for the GP lightcurve noise models (one dimensional).
"""

# TODO: a base GP object would require big changes overall,
# check back later. For now, will only have a 1D detrender as a base.

# class BaseGPModel(object):
#     """Contains the george.GP object and optimisation/detrending.
#     TODO: should also contain priors and kernel somehow no?

#     Constituents:
#     GP:                 self.gp
#     kernel:             
#     additive model:     self._model

#     State:
#     hp:                 the current state of the hyperparameters
#     ts_basis:          the last calculated basis
#     """
#     pass

import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg.linalg import LinAlgError
from scipy.optimize import (minimize, differential_evolution,
                            fmin_powell, OptimizeResult)

import george

from . import model_objects
from .kernels import PriorInitialisationError
from .. import vislib, lc_preparation, lc_utils

# ---------------------------------------
# One-dimensional detrender object (base)
# ---------------------------------------

# TODO: lcf -> ts (timeseries) naming convention, though alias
# object.ts as object.lcf

class LCNoiseModel1D(object):
    """Contains the george.GP object, priors, optimisation & detrending.

    At the level of this object and its internals, the GP
    hyperparameters are known only in the george vector parametrisation.
    Conversion from this parametrisation to something more intuitive
    is only done in the constituent kernel/prior objects.

    Attributes:
        _kernel_type (str): 'SemiPeriodic1D', 'Classic1D'
        _hp_names (list): actual names of the hyperparameters, read
            from a kernel object
        _X_cols (list): name(s) of the columns in ts that hold the
            independent parameter(s)
        _model_cols (list or str): names(s) of the columns in ts that
            hold the independent parameter of the additional model
        _vector_dim (int): number of hyperparameters TODO
        _hp_names_local (list of str): local names of the parameters
        _hp_names_gp (list of str): names as they appear in GP object
        ts (pd.DataFrame): contains a stored full timeseries which is
            used to select the bases, optimise the hyperparameters, and
            as the default for prediction.
        ts_basis (pd.DataFrame): stored basis timeseries, with outliers
            removed, and potentially with sub-sampling. Detrending and
            noise components are calculated from this time-series.
            Setter/getters: get_basis, set_basis.
        hp_vector (array): cached hyperparameter values; used as the
            default when no hyperparameter values are selected.
            Values of the hp_vector that are produced by optimization
            are stored here.
        ocut (float): stored value of outlier clipping factor

    Constituents:
        LCKernel: user-defined object that holds the kernel, prior,
            domain, as well as parameter transformations.
        kernel (george.kernels.Kernel): the george.kernel object, which
            is itself read from LCKernel
        gp (george.GP): the george object that actually performs the
            matrix operations and likelihood calculation
        add_model (model_objects.BaseModel): additional model, with its
            own parameters, that is added linearly with the GP noise
            models

    Aliased:
        ts -> lcf
        ts_basis -> lcf_basis
    """

    # These are always local parameters; they represent the white
    # noise in the lightcurve (potentially can be set to 0)
    _local_parameters = ('2ln_wn',)

    # Name changes overall: self._kernel -> self.LCKernel
    # self._X_cols > self._X_cols
    # self.model_col -> self._model_cols
    # self.get_full_prediction -> predict_trend

    def __init__(self, ts, kernel_class, additional_model='y_offset',
                 X_cols=None, model_cols='t', infer_wn=True,
                 ocut=5, N_opt=1000, N_gp=None):
        """Assigns a base lightcurve and kernel to the GP object.

        Args:
            ts (pd.DataFrame): the dependent parameter is always 'f',
                the rest are specificed in X_cols and model_cols
            kernel_class (LCKernel): contains the george.kernel, and the
                hyperparameter priors, domains, and transformations
            additional_model (model_objects.BaseModel='y_offset'):
                specifies the additive model to be used. By default,
                will just be a frozen y_offset. Choose from following:
                - 'y_offset' estimates y_offset and puts it in GP.
                    This is not a free parameter by default
                - None same as y_offset
                - 'ramp' sets up a RampModel, with y_offset etc...
                - 1.0 or an numeric values forces that mean
                - otherwise give it a direct model object as defined in
                    .model_objects.BaseModel
            X_cols: name of the columns in ts that hold the
                independent parameter(s)
            model_cols (list or str): name(s) of the columns in ts that
                hold the independent parameter of the additional model
            infer_wn (bool=True): to calculate a first estimate
                for the white noise hyperparamter. Without this,
                the first outlier mask will be wrong
            ocut (float=5): the outlier cutting factor
            N_opt (int=1000): number of points to use for hyperparameter
                optimization, by default; if None, then it will use all
                the points available by default.
            N_opt (int=None): number of points to use for GP regression,
                by default; if None, then it will use all the points
                available by default.
        """

        if (N_gp is not None and N_gp > len(ts)) \
            or (N_opt is not None and N_opt > len(ts)):
            warnings.warn("N_gp or N_opt are larger than the number "
                          "of points in the timeseries. Cutting down "
                          "to len(ts).")

        # Set up kernel, prior, and hyperparameters
        self.LCKernel = kernel_class
        self._kernel = kernel_class.kernel
        self._kernel_type = str(type(kernel_class))
        self.kernel_lnprior = kernel_class.log_prior

        self._X_cols = list(self.LCKernel.default_X_cols) if X_cols is None \
                                                          else list(X_cols)
        self._model_cols = model_cols
        self.N_opt = None if N_opt is None or N_opt > len(ts) else N_opt
        self.N_gp = None if N_gp is None or N_gp > len(ts) else N_gp

        if isinstance(self._model_cols, list) and len(self._model_cols) == 1:
            self._model_cols = self._model_cols[0]

        # if X_cols is None:
        # 	self._X_cols = list(self.LCKernel.default_X_cols)
        # else:
        # 	self._X_cols = list(X_cols)

        # Prepare the model
        if additional_model is None or additional_model == 'y_offset':
            # If this is the case, then the y_offset is taken care of
            # within the george.gp object
            self._model = None
        elif additional_model == 'ramp':
            y_offset = np.nanmedian(ts['f'])
            self._model = model_objects.RampModel(lcf=ts,
                                                  y_offset=y_offset,
                                                  x_ndim=len(self._X_cols))
        elif not isinstance(additional_model, str):
            self._model = additional_model
        else:
            raise NotImplementedError("additional_model {} not recognised.".format(additional_model))

        # Initialise the george.GP object
        if additional_model is None or additional_model == 'y_offset':
            self.gp = george.GP(kernel=self._kernel,
                                white_noise=1.0,
                                mean=np.nanmedian(ts['f']),
                                fit_white_noise=True,
                                fit_mean=True)
        elif isinstance(additional_model, (int, float)):
            self.gp = george.GP(kernel=self._kernel,
                                white_noise=1.0,
                                mean=additional_model,
                                fit_whitetsn=True)
        else:
            self.gp = george.GP(kernel=self._kernel,
                                white_noise=1.0,
                                mean=self._model,
                                fit_white_noise=True,
                                fit_mean=True)

        # Model parameters and names
        if additional_model in ('y_offset', None) or isinstance(additional_model, (int, float)):
            self._model_parameters = ('y_offset',)
            self._model = None
        else:
            self._model_parameters = tuple(
                self._model.get_parameter_names(include_frozen=True))

        self._hp_names_local = self._model_parameters + \
                               self._local_parameters + \
                               self.LCKernel.parameter_names

        self._hp_names_gp = self.gp.parameter_names

        if np.shape(self._hp_names_local) != np.shape(self._hp_names_gp):
            raise ValueError("Internal and GP parameter shape doesn't match.")

        # White noise parameter bounds are separate (on 2ln_wn)
        self.wn_bounds = [-20.0, 0.0]

        # Set initial white noise
        if infer_wn:
            self.set_parameter(
                name='2ln_wn', local_name=True,
                value=lc_preparation.get_white_noise(ts, chunk_size=10))

        # Freeze mask doesn't need to be set; it's already in gp
        # Freeze the y_offset parameter in all default cases
        if additional_model in ('ramp', None) or isinstance(additional_model, (int, float)):
            self.freeze_parameter('y_offset')

        # Set-up the internal timeseries (detached from input)
        self.set_ts(ts)

        # ib is the initial_basis
        ib = ts[~lc_preparation.mask_flares(ts.f)]
        ib = ib.assign(opt_basis=False, gp_basis=False)

        if N_opt is not None and N_opt < len(ib):
            opt_idx = np.random.choice(ib.index, N_opt, replace=False)
            ib.loc[opt_idx, 'opt_basis'] = True
        else:
            ib['opt_basis'] = True

        if N_gp is not None and N_gp < len(ib):
            gp_idx = np.random.choice(ib.index, N_gp, replace=False)
            ib.loc[gp_idx, 'gp_basis'] = True
        else:
            ib['gp_basis'] = True

        self.set_basis(ib)

        # Set internal values
        self.set_ocut(ocut)
        self.__detrending_hp = None		# Used to cache the latest hp
                                        # that were used to detrend the
                                        # saved lightcurve. Done so that
                                        # we can check if it's detrended

        self.OversamplingPopulationError = OversamplingPopulationError
        self.NonPositiveDefiniteError = NonPositiveDefiniteError
        self.PriorInitialisationError = PriorInitialisationError

        X_0 = self.get_basis(only='gp_basis')[self._X_cols].values
        self.compute(X_0)

    # Basic method/decorators of GP
    # -----------------------------

    def compute(self, X_array, *args, **kwargs):
        """Decorator around gp.compute to handle Exceptions."""

        try:
            self.gp.compute(X_array, *args, **kwargs)
        except LinAlgError:
            raise NonPositiveDefiniteError(
                message='hp_dict = {}'.format(self.get_parameter_dict()),
                X=X_array, hp=self.hp)

    def predict(self, y_basis, X_predict, *args, **kwargs):
        """Decorator around gp.predict to handle Exceptions."""

        try:
            return self.gp.predict(y_basis, X_predict, *args, **kwargs)
        except Exception as e:
            # These are generally LAPACKError-s
            raise LAPACKError(
                message="LAPACKError",
                exception=e,
                X_predict=X_predict,
                y_basis=y_basis,
                X_basis=self.gp._x,
                hp=self.get_parameter_dict(include_frozen=True))

    # Regression methods
    # ------------------

    def predict_trend(self, X_predict=None, hp=None):
        """Predicts the full model trend: including internal model.

        Args:
            X_predict
            hp
        """

        if hp is not None:
            self.set_hp(hp)
        if X_predict is None:
            #t_predict = self.get_ts()[self._model_cols].values
            X_predict = self.get_ts()[self._X_cols].values
        elif isinstance(X_predict, pd.DataFrame):
            # t_predict = X_predict[self._model_cols].values
            X_predict = X_predict[self._X_cols].values
        elif isinstance(X_predict, np.ndarray):
            pass
        else:
            raise NotImplementedError("X_predict needs to be a DataFrame.")

        y_basis = self.get_basis(only='gp_basis')['f'].values
        X_basis = self.get_basis(only='gp_basis')[self._X_cols].values

        self.compute(X_basis)

        f_predict, _ = self.predict(y_basis, X_predict)

        return f_predict

    def detrend(self, ts_predict=None, hp=None, save=True):
        """Produces a detrended timeseries.

        Args:
            ts_predict (pd.DataFrame): the points on which to predict
                the trend. Contains 'f', X_cols, model_cols.
                Default: self.ts
            ts_basis (pd.DataFrame): the points to use as the
                regression basis. Contains 'f', X_cols, model_cols.
                Default: self.ts_basis
            hp (1D array): vector for the GP hyperparameters
                Default: self.hp
            save (bool): if True, saves the ts_predict to self.ts
                Default: False

        Returns:
            ts_predict (pd.DataFrame): detrended values contained in
                'f_detrended'; with trend in 'f_gp_trend'
        """

        if ts_basis is None:
            ts_basis = self.get_basis(only='gp_basis')
        if ts_predict is None:
            ts_predict = self.ts.copy()
        if hp is not None:
            # Need to set the hp so that self._model works
            self.hp = hp

        if self._model is not None:
            f_model = self._model.get_value(ts_predict[self._model_cols])
        else:
            f_model = self.get_parameter('y_offset')

        y_trend = self.predict_trend(ts_predict)

        ts_predict['f_model'] = f_model
        ts_predict['f_trend_gp'] = y_trend - f_model

        # Special behaviour: ramp is supposed to be removed (detrender),
        # while other models aren't. So if the model is ramp, then
        # remove from detrended, otherwise keep in detrended.
        # i.e. in all cases, keep the y_offset in detrended.
        y_detrended = ts_predict['f'] - y_trend

        if self._model is None:
            y_detrended = y_detrended + self.get_parameter('y_offset')
        elif isinstance(self._model, model_objects.RampModel):
            y_detrended = y_detrended + self.get_parameter('y_offset')
        else:
            y_detrended = y_detrended + self._model.get_value()

        ts_predict['f_detrended'] = y_detrended

        if save:
            self.ts = ts_predict
            self.__detrending_hp = self.hp

        if 'opt_basis' in ts_predict or 'gp_basis' in ts_predict:
            warnings.warn('opt_basis and gp_basis in ts_predict.')

        return ts_predict #.drop(['opt_basis', 'gp_basis'])

    def mask_outliers(self, ts=None, ocut=None, hp=None):
        """Masks points further than ocut from the mean.

        Args:
            ts (pd.DataFrame): the points from which to mask outliers.
                Must contain columns 'f' and X_cols.
                Default: self.ts.
            ts_basis (pd.DataFrame): the points to use as the basis
                from which to regress the trend. Must contain columns
                'f' and X_cols.
                Default: self.ts_basis.
            ocut (float): the number of sigmas that determine an outlier.
            hp (1D array): vector for the GP hyperparameters.
                Default: self.hp

        Returns:
            outlier_mask (1D array of bool): True if point is an outlier.
        """
        if ts is None:
            ts = self.ts
        if hp is None:
            hp = self.hp
        if ocut is None:
            ocut = self.get_ocut()

        f = ts['f'].values
        X = ts[self._X_cols].values

        mu = self.predict_trend(X_predict=ts)
        std = np.exp(self.get_parameter('2ln_wn')/2)

        outlier_mask = (f > (mu + ocut*std)) | (f < (mu - ocut*std))

        return outlier_mask

    def select_basis(self, N_opt=None, N_gp=None, ts=None,
                     cut_outliers=True, cut_transits=True, save=True,
                     save_o_flag=True, quiet=True, cut_flares=False,
                     **mask_kwargs):
        """Removes outliers and samples an ts_basis from a lightcurve.

        The ts_basis has an column called 'sub_flag', determined by N.
        The basis are the points that have sub_flag = True; this
        determines the subset of the basis which is used for
        optimization. When retrieving the basis, to ask for
        the full basis or the subset, use .get_basis(only=...)

        Args:
            N (int, optional): number of points to use.
            ts (pd.Dataframe, optional): uses self._ts by default
            ts_basis (pd.DataFrame, optional): by default full self._ts.
                Actually, this is swallowed by **mask_kwargs.
            cut_outliers (bool): if False, doesn't remove outliers.
            cut_transits (bool): if False, doesn't remove transit
                points, taken from the 't_flag' column in ts.
            save (bool=True): if True, saves to self._basis
            save_o_flag (bool=True): updates the outliers in the main
                saves timeseries (.ts)
            quiet (bool): whether to throw an exception if N > len(ts)
            cut_flares (bool): removes all flare like points that
                are massive outliers above when doing the first
                detrend.
            **mask_kwargs: ocut, hp, ts_basis

        Returns:
            ts_basis (pd.DataFrame): also saved to self._ts_basis
        """

        # TODO: temporary to alias vs previous versions
        if 'N' in mask_kwargs:
            N = mask_kwargs.pop('N')
            N_opt = N if N_opt is None else N_opt

        if ts is None:
            ts = self.get_ts()
        if cut_flares:
            ts = ts[~lc_preparation.mask_flares(ts.f)]
        if N_opt is None:
            N_opt = self.N_opt
        if N_gp is None:
            N_gp = self.N_gp

        # Checks before we start with the long calculations
        if not None in (N_gp, N_opt) and max(N_opt, N_gp) > len(ts) \
            and not quiet:
            raise OversamplingPopulationError

        # Do this first so it's not part of any subsets
        if cut_transits and 't_flag' in ts.columns:
            ts = ts[~ts.t_flag]

        if cut_outliers:
            outlier_mask = self.mask_outliers(ts=ts, **mask_kwargs)
        else:
            outlier_mask = np.ones(len(ts), dtype=bool)

        ts_basis = ts.copy()[~outlier_mask]

        # NOTE: variable ts may have been modified by this point

        # Checks before we start with the long calculations
        if N_opt is None:
            opt_idx = ts_basis.index
        elif N_opt > len(ts_basis) and quiet:
            warnings.warn("basis set will be undersampled. N_opt "
                          "is bigger than the length of ts_basis.")
            opt_idx = ts_basis.index
        elif N_opt <= len(ts_basis):
            opt_idx = opt_idx = np.random.choice(
                ts_basis.index, N_opt, replace=False)
        else:
            raise OversamplingPopulationError

        if N_gp is None:
            gp_idx = ts_basis.index
        elif N_gp > len(ts_basis) and quiet:
            warnings.warn("basis set will be undersampled. N_gp "
                          "is bigger than the length of ts_basis.")
            gp_idx = ts_basis.index
        elif N_gp <= len(ts_basis):
            gp_idx = gp_idx = np.random.choice(
                ts_basis.index, N_gp, replace=False)
        else:
            raise OversamplingPopulationError

        ts_basis = ts_basis.assign(opt_basis=False, gp_basis=False)
        ts_basis.loc[opt_idx, 'opt_basis'] = True
        ts_basis.loc[gp_idx, 'gp_basis'] = True

        if save:
            self.set_basis(ts_basis)
        if save_o_flag and cut_outliers:
            self.set_ts(ts.assign(o_flag=outlier_mask))

        return ts_basis

    # Probabilistic methods
    # ---------------------

    def log_likelihood(self, hp_vector=None, f=None):
        """Returns the log_likelihood of the GP model.

        Updates the internal hp automatically.

        NOTE: needs the f=None argument, it makes more sense to choose
        basis once while setting up optimisation problem, and then
        entering it through the optimiser args.

        Args:
            hp_vector (1 array): values to use for the hyperparameters.
            f (1D array): the flux points where gp was last computed.
        """

        if hp_vector is None:
            hp_vector = self.hp
        else:
            self.set_hp(hp_vector)

        if f is None:
            f = self.get_basis(only='opt_basis').f

        return self.gp.log_likelihood(f, quiet=True)

    def grad_log_likelihood(self, hp_vector=None, f=None):
        """Returns the gradient of the log_likelihood of the GP model.

        Updates the internal hp automatically.

        NOTE: needs to f=None argument, it makes more sense to choose
        basis once while setting up optimisation problem, and then
        entering it through the optimiser args.

        Args:
            hp_vector (1 array): values to use for the hyperparameters.
            f (1D array): the flux points where gp was last computed.
        """

        if hp_vector is None:
            hp_vector = self.hp
        else:
            self.hp = hp_vector

        if f is None:
            f = self.get_basis(only='opt_basis').f

        return self.gp.grad_log_likelihood(f, quiet=True)

    def log_prior(self, hp_vector=None):
        """Calculates the log_prior of the constituent models.
        Updates the internal hp automatically."""

        if hp_vector is None:
            hp_vector = self.hp
        else:
            self.set_hp(hp_vector)

        return (self.LCKernel.log_prior() \
              + self.log_prior_wn() \
             + (self._model.log_prior() if self._model is not None else 0))

    def log_prior_wn(self, value=None):
        """Calculates the log prior in the wn hyperparameter."""

        value = self.get_parameter('2ln_wn') if value is None else value

        if value < self.wn_bounds[0] or value > self.wn_bounds[1]:
            return -np.inf
        else:
            return 0.0

    def lnposterior(self, hp_vector=None, f=None):
        """Returns the posterior of the GP model at hp_vector.

        Updates the internal hp automatically.

        Args:
            hp_vector (1 array): values to use for the hyperparameters.
            f (1D array): the flux points where gp was last computed.
        """

        if hp_vector is None:
            hp_vector = self.hp
        else:
            self.set_hp(hp_vector)

        if f is None:
            f = self.ts_basis.f

        return self.log_likelihood(hp_vector, f) + self.log_prior(hp_vector)

    def neg_lnposterior(self, hp_vector=None, f=None):
        """Negative of the lnposterior, for minimisation."""

        return -self.lnposterior(hp_vector, f)

    def grad_lnposterior(self, hp_vector=None, f=None):
        """Returns the gradient of the lnposterior w.r.t hp_vector."""
        # TODO: implement the prior gradients and this
        raise NotImplementedError

    def neg_grad_lnposterior(self, hp_vector=None, f=None):
        """Negative gradient of lnposterior, for minimisation."""

        return -self.grad_lnposterior(hp_vector, f)

    # Optimization and inference methods
    # ----------------------------------

    def optimise_hp_powell(self, ts=None, initial_hp=None, diagnose=False, save=True, **fmin_args):
        """Minimises the posterior w.r.t hyperpameters with fmin_powell.

        Args:
            ts (pd.DataFrame): the lightcurve to use as basis
                for optimization (default: self._ts_basis)
            initial_hp (array, Optional): the initial points
                for optimization
            diagnose (bool): if True, prints and returns full output
            save (bool): whether to save result into hp
                Cannot be done if diagnose is True.
            **fmin_args (dict): other inputs into the minimise function
                choices: 'full_output', 'disp', 'retall', 'direc',
                'xtol', 'ftol', 'maxiter', 'maxfun'

        Returns:
            result (1D array): the output from fmin_powell, may be
                more complicated if diagnose==True.
        """

        if ts is None:
            ts = self.get_basis(only='opt_basis')
        if initial_hp is None:
            initial_hp = self.hp
        if diagnose:
            for key in ('full_output', 'retall', 'disp'):
                fmin_args[key] = True
        else:
            for key in ('full_output', 'retall', 'disp'):
                fmin_args[key] = False

        f = ts['f'].values
        X = ts[self._X_cols].values

        self.compute(X)
        result = fmin_powell(self.neg_lnposterior, x0=initial_hp, args=(f,), **fmin_args)

        if not diagnose:
            if isinstance(result, OptimizeResult) and save:
                self.set_hp(result.x)
            elif save:
                self.set_hp(result)
        elif diagnose:
            if self.neg_lnposterior() != result[1][1]:
                print("Posterior mismatch.")
                print("Current value:", self.neg_lnposterior())
                print("Optimised value:", result[1][1])

        return result

    def optimise_hp_grad(self, method='BFGS', ts=None, initial_hp=None,
                         save=True, **min_args):
        """Minimise posterior w.r.t hyperpameters with a gradient method.

        Args:
            method (str): 'BFGS', etc...
                Currently, gradients are not possible
            ts (pd.DataFrame): the lightcurve to use as basis
                for optimization (default: self._ts_basis)
            initial_hp (array, Optional): the initial points
                for optimization
            save (bool): whether to save result into hp
                Cannot be done if diagnose is True.
            **min_args (dict): other inputs into the minimise function

        Returns:
            ...
        """

        if ts is None:
            ts = self.get_basis(only='opt_basis')
        if initial_hp is None:
            initial_hp = self.hp

        f = ts['f'].values
        X = ts[self._X_cols].values

        self.compute(X)

        result = minimize(self.neg_lnposterior,
                          x0=initial_hp,
                          args=(f,),
                          method=method,
                          jac=self.neg_grad_lnposterior,
                          **min_args)

        # TODO: Check the form of result first, maybe take result.x
        print("First run of grad optimizer, check result.")
        import pdb; pdb.set_trace()

        if save:
            self.set_hp(result)

        return result

    def evolve_hp(self, ngen=50, npop=100, ts=None, save=True, full_result=False):
        """Estimates a posterior best-fit w.r.t hyperparameters with DE.*

        *Differential Evolution

        Args:
            ngen (int)
            npop (int)
            ts (pd.DataFrame, Optional): default is **ts_basis**;
                i.e the lightcurve on which we want to optimize the
                hyperparameters
            save (bool): to save hp into hp (default: True)
            full_result (bool): return the full scipy.OptimizationResult
                or just the best fit vector (default: False)


        Returns:
            The OptimizeResult object from scipy.optimize, where
            result.x contains the optimized values, and .success
            and .message contain the successful exit flag and any
            exit messages respectively.
        """

        if ts is None:
            ts = self.get_basis(only='opt_basis')

        f = ts['f'].values
        X = ts[self._X_cols].values
        de_bounds = self.de_bounds

        self.compute(X)

        result = differential_evolution(self.neg_lnposterior,
                                        bounds=de_bounds,
                                        popsize=npop,
                                        maxiter=ngen,
                                        args=(f,))

        if save:
            self.set_hp(result.x)
        if full_result:
            return result
        else:
            return result.x

    # Utilities, analysis, visualisation
    # ----------------------------------

    def plot_model(self, ts=None, show=False, title=None):
        """Plots the model, using `k2gp.analysis.plot_model`."""

        raise NotImplementedError("Not implemented for 1D case.")

        if ts is None:
            ts = self._ts

        if 'f_detrended' in ts:
            vislib.plot_gp_model(lcf=ts, show=show, title=title)
        else:
            raise AttributeError("f_detrended is not in the lightcurve.")

    def plot_raw(self, ts=None, show=False, title=None):
        """Just plots the raw data in f."""

        if ts is None:
            ts = self.ts

        fig, ax = plt.subplots(2)
        ax[0].plot(ts.t, ts.x, 'r.', ts.t, ts.y, 'b.')
        ax[1].plot(ts.t, ts.f, 'k.')
        fig.suptitle(title)

        if show:
            plt.show()
        else:
            fig.show()

    def calc_cdpp(self, columns='f_detrended', ts=None,
                  remove_outliers=False, remove_transits=True):
        """Calculates the CDPP in columns.

        Args:
            columns (tuple/str): which columns to calculate the cdpp
                for, will return a dictionary unless a single
                column is specified.
            ts (pd.DataFrame, optional)
            remove_outliers (bool, False): masks o_flag values
            remove_transits (bool, True): masks t_flag values

        Returns:
            dict() of cdpp's with keys = columns, or a single float
            if a single column is specified
        """

        ts = ts if ts is not None else self._ts
        if remove_outliers:
            ts = ts[~ts.o_flag]
        if remove_transits and 't_flag' in ts.columns:
            ts = ts[~ts.t_flag]

        if columns == 'all':
            columns = ('f', 'f_detrended')
        elif isinstance(columns, str):
            columns = (columns,)

        cdpps = dict()
        for col in columns:
            cdpps[col] = lc_utils.calc_cdpp(ts, column=col)

        if len(columns) == 1:
            return cdpps[columns[0]]
        else:
            return cdpps

    def verify_prior(self, vector=None, quiet=False,
                     include_frozen=True, info_str=None):
        """Checks if the vector is inside the prior bounds.

        Args:
            vector (np.array): optional
            quiet (bool): if True, will only return a False if
                verification fails, otherwise raises a full error
            include_frozen (bool): if True, vector is only the
                unfrozen parameters
            info_str (str): str to append to error message if
                verification fails
        """

        initial_vector = np.copy(self.get_hp(include_frozen=True))

        if vector is None:
            vector = np.copy(self.get_hp(include_frozen=include_frozen))
        self.set_hp(vector, include_frozen=include_frozen)

        if np.isfinite(self.log_prior()):
            self.set_hp(initial_vector, include_frozen=True)
            return True
        elif quiet:
            self.set_hp(initial_vector, include_frozen=True)
            return False
        else:
            raise PriorInitialisationError((
                "Initial hyperparameter values are out of prior bounds.\n"
                "hp: {}\n"
                "bounds: {}\n"
                "info_str: {}"
                "".format(self.get_hp(True), self.parameter_bounds, info_str)))


    # Internal parameter properties
    # -----------------------------

    def __len__(self):
        return len(self.gp)

    @property
    def _hp_active(self):
        return self.gp.unfrozen_mask
    unfrozen_mask = _hp_active

    def get_hp(self, *args, **kwargs):
        return self.gp.get_parameter_vector(*args, **kwargs)

    def set_hp(self, *args, **kwargs):
        self.gp.set_parameter_vector(*args, **kwargs)

    hp = property(get_hp, set_hp)

    def get_parameter_names(self, *args, **kwargs):
        return self.gp.get_parameter_names(*args, **kwargs)

    @property
    def parameter_names(self):
        return self.gp.parameter_names

    @parameter_names.setter
    def parameter_names(self, vector):
        self.gp.parameter_names = vector

    @property
    def parameter_vector(self):
        return self.gp.parameter_vector

    @parameter_vector.setter
    def parameter_vector(self, vector):
        self.gp.parameter_vector = vector

    def get_parameter_dict(self, include_frozen=False, local_names=True):
        """Returns the parameter dict, local_names=True by default."""

        names = self.get_parameter_names(include_frozen)
        if local_names:
            names = self._pname_to_local(names)
        return OrderedDict(zip(names, self.get_hp(include_frozen)))

    def get_ocut(self):
        return self._ocut

    def set_ocut(self, value):
        if value <= 0:
            raise ValueError("Invalid ocut value, must be greater than zero.")
        self._ocut = value

    ocut = property(get_ocut, set_ocut)

    @property
    def de_bounds(self):
        """Returns the de_bounds on the currently active parameters.
        
        These are the differential evolution bounds.
        """

        raise NotImplementedError

        # ramp_bounds = list(self._ramp.get_parameter_bounds())
        # kernel_bounds = []
        # for name in self.get_parameter_names():
        # 	key = self._pname_to_local(name)
        # 	if key in self._kernel.parameter_names:
        # 		kernel_bounds.append(self._kernel.de_domain[key])
        # return ramp_bounds + kernel_bounds

    # Internal parameter methods
    # --------------------------

    def _convert_name_to_gp(self, name):
        """Converts the parameter name from local to gp."""

        # The indices corresponding (ordered)
        if not hasattr(name, '__len__') or isinstance(name, str):
            return self._hp_names_gp[self._hp_names_local.index(name)]
        else:
            return tuple(self._hp_names_gp[self._hp_names_local.index(n)]
                         for n in name)

    def _pname_to_local(self, name):
        """Converts the parameter name from gp to local."""

        # The indices corresponding (ordered)
        if not hasattr(name, '__len__') or isinstance(name, str):
            return self._hp_names_local[self._hp_names_gp.index(name)]
        else:
            return tuple(self._hp_names_local[self._hp_names_gp.index(n)]
                         for n in name)

    def get_parameter(self, name, local_name=True):
        """Gets a specific parameter; name is local by default."""
        name = self._convert_name_to_gp(name) if local_name else name
        return self.gp.get_parameter(name)

    def set_parameter(self, name, value, local_name=True):
        """Sets a specific parameter; name is local by default."""
        name = self._convert_name_to_gp(name) if local_name else name
        self.gp.set_parameter(name, value)

    def freeze_parameter(self, name, local_name=True):
        """By default the name is local."""
        name = self._convert_name_to_gp(name) if local_name else name
        if isinstance(name, str):
            self.gp.freeze_parameter(name)
        else:
            for n in name:
                self.gp.freeze_parameter(n)

    def thaw_parameter(self, name, local_name=True):
        """By default the name is local."""
        name = self._convert_name_to_gp(name) if local_name else name
        if isinstance(name, str):
            self.gp.thaw_parameter(name)
        else:
            for n in name:
                self.gp.thaw_parameter(n)

    def freeze_model(self):
        """Freezes the hyperparameters for the ramp."""
        self.freeze_parameter(self._model_parameters)

    def thaw_model(self):
        """Activates the hyperparameters for the ramp."""
        self.thaw_parameter(self._model_parameters)

    @property
    def model_bounds(self):
        if self._model is not None:
            return self._model.parameter_bounds
        else:
            return [[-20, 20]]

    @property
    def parameter_bounds(self):
        """Fused the various bounds into one array."""
        return np.concatenate([self.model_bounds,
                               [self.wn_bounds],
                               self.LCKernel.bounds])

    # TODO: make a setter function so we can change the bounds
    def get_parameter_bounds(self, include_frozen=False):
        return self.parameter_bounds[self.unfrozen_mask]

    def get_bound_dict(self, local_names=True):
        if local_names:
            return dict(zip(self._hp_names_local, self.parameter_bounds))
        else:
            return dict(zip(self._hp_names_gp, self.parameter_bounds))

    @property
    def is_detrended(self):
        """Returns True if saved ts has been detrended
        with the most recent parameters."""
        if self.hp == self.__detrending_hp:
            if 'f_detrended' in self._ts.columns:
                return True
            else:
                warnings.warn("The current lightcurve has no detrending "
                              "column, but is flagged as 'is_detrended' "
                              "with the current hyperparameters.")
                return False
        else:
            return False

    # Lightcurves getters and setters
    # -------------------------------
    def get_ts(self, clean=False):
        if not clean:
            return self._ts.copy()
        else:
            return self._ts.drop(columns=['opt_basis', 'gp_basis'])

    def set_ts(self, ts):
        if not pd.Series(['f'] + self._X_cols).isin(ts.columns).all():
            raise ValueError("The ts must contain 'f' and X_cols.")
        self._ts = ts.copy()

    def get_basis(self, only=None, full=None):
        """Gives a subset basis.
        
        Arguments:
            only
            full (bool=None): for back-compatibility, if True/False,
                will return N_gp/N_opt
        """
        if only == 'gp_basis':
            return self._ts_basis[self._ts_basis.gp_basis].copy()
        elif only == 'opt_basis':
            return self._ts_basis[self._ts_basis.opt_basis].copy()
        elif not full and full is not None and only is None:
            # TODO: Deprecated but for back-compatibility
            return self._ts_basis[self._ts_basis.opt_basis].copy()
        elif full and only is None:
            # TODO: Deprecated but for back-compatibility
            return self._ts_basis[self._ts_basis.gp_basis].copy()
        elif only is None:
            return self._ts_basis.copy()
        else:
            raise ValueError("Value for argument 'only' not recognised: "
                             "{}".format(only))

    def set_basis(self, ts_basis):
        if not pd.Series(['f'] + self._X_cols).isin(ts_basis.columns).all():
            raise ValueError("The ts must contain 'f' and X_cols.")
        if 'N_opt' not in ts_basis.columns:
            warnings.warn("Attempting to set a basis with no 'N_opt'"
                          "subset.")
        if 'N_gp' not in ts_basis.columns:
            warnings.warn("Attempting to set a basis with no 'N_gp'"
                          "subset.")
        self._ts_basis = ts_basis.copy()

    ts = property(get_ts, set_ts)
    # ts_basis will not have gp_basis or opt_basis subselected
    ts_basis = property(get_basis, set_basis)

    # Aliases to allow more seamless integration with previous
    # --------------------------------------------------------
    lcf = ts
    lcf_basis = ts_basis
    detrend_lightcurve = detrend

    get_lcf = get_ts
    set_lcf = set_ts


# For 3D, add component methods, modify predict_trend, detrend
# Potentially re-define aliases

# ----------
# Exceptions
# ----------

class OversamplingPopulationError(ValueError):
    """Thrown when attempting to sample more values than there
    exist in a set, without replacement."""

    pass

class NonPositiveDefiniteError(LinAlgError):
    """When the GP coordinate is non positive definite.

    i.e to catch the LinAlgError throw by gp.compute(X).

    Attributes:
        X: the """

    def __init__(self, X, message=None, hp=None):

        message = message if message is not None else ''
        super().__init__('NonPositiveDefiniteError - ' + message)
        self.X_basis = pd.DataFrame(X, columns=['x', 'y', 't'])
        self.hp = hp 

    def attach_detrender(self, k2_detrender_class):
        """Setter for packing the whole detrender class."""

        self.k2_detrender = k2_detrender_class

class LAPACKError(Exception):
    """When the GP coordinate is non positive definite.

    i.e to catch the LinAlgError throw by gp.compute(X).

    Attributes:
        X: the """

    def __init__(self, X_predict, y_basis, exception, X_basis=None,
                 message=None, hp=None):

        message = message if message is not None else ''
        message = ("{}\nReceived error message: {}"
                   "".format(message, str(exception)))

        super().__init__('LAPACKError - possible _flapack.error - ' + message)

        self.X_predict = pd.DataFrame(X_predict, columns=['x', 'y', 't'])
        self.f_predict = pd.Series(y_basis)
        self.exception = exception

        if X_basis is not None:
            self.X_basis = pd.DataFrame(X_basis, columns=['x', 'y', 't'])
        if hp is not None:
            self.hp = hp