"""GP models for long lightcurves (Kepler, TESS, K2 short-cadence)."""
import warnings

import numpy as np
import pandas as pd
from numpy.linalg.linalg import LinAlgError

import george

from . import model_objects
from .gp_model_base import (LCNoiseModel1D,
                            NonPositiveDefiniteError,
                            OversamplingPopulationError,
                            PriorInitialisationError,
                            LAPACKError)
from .gp_model_3d import LCNoiseModel3D
from .. import lc_preparation


class LLCNoiseModel1D(LCNoiseModel1D):
    """GP noise model for long lightcurves."""

    pass

# Main changes:
# N_opt, N_gp for optimisation and detrending basis defaults
# basis full removed; always enter N_opt and N_gp
#
# Generally will need to separately set/sample the basis;
# it is not done automatically by the methods that need it.
# This is also where we pick the basis size.
#
#
# In terms of usage; figured out it's better if we can enter ts in
# most prediction/detrending/regression methods, but NOT basis; basis
# must be entered "manually" with set_basis or select_basis.
# TODO: move this change to all detrender objects.


class LLCNoiseModel3D(LCNoiseModel3D):
    """GP noise model for long lightcurves."""

    def __init__(self, ts, *args, N_opt=2500, N_gp=10000, **kwargs):
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

        super().__init__(ts, *args, N_opt=N_opt, N_gp=N_gp, **kwargs)
        self.N_split = 1000     # size of chunks when doing regression

    # def __init__(self, ts, kernel_class, additional_model='y_offset',
    #              X_cols=None, model_cols='t',
    #              infer_wn=True, ocut=5, N_opt=2500, N_gp=6000):
    #     if N_gp > len(ts) or N_opt > len(ts):
    #         warnings.warn("N_gp or N_opt are larger than the number "
    #                       "of points in the timeseries. Cutting down "
    #                       "to len(ts).")

    #     # Set up kernel, prior, and hyperparameters
    #     self.LCKernel = kernel_class
    #     self._kernel = kernel_class.kernel
    #     self._kernel_type = str(type(kernel_class))
    #     self.kernel_lnprior = kernel_class.log_prior

    #     self._X_cols = list(self.LCKernel.default_X_cols) if X_cols is None \
    #                                                       else list(X_cols)
    #     self._model_cols = model_cols
    #     self.N_opt = N_opt if N_opt > len(ts) else len(ts) - 1
    #     self.N_gp = N_gp if N_gp > len(ts) else len(ts) - 1
    #     self.N_split = 1000     # size of chunks when doing regression

    #     if isinstance(self._model_cols, list) and len(self._model_cols) == 1:
    #         self._model_cols = self._model_cols[0]

    #     # if X_cols is None:
    #     # 	self._X_cols = list(self.LCKernel.default_X_cols)
    #     # else:
    #     # 	self._X_cols = list(X_cols)

    #     # Prepare the model
    #     if additional_model is None or additional_model == 'y_offset':
    #         # If this is the case, then the y_offset is taken care of
    #         # within the george.gp object
    #         self._model = None
    #     elif additional_model == 'ramp':
    #         y_offset = np.nanmedian(ts['f'])
    #         self._model = model_objects.RampModel(lcf=ts,
    #                                               y_offset=y_offset,
    #                                               x_ndim=len(self._X_cols))
    #     elif not isinstance(additional_model, str):
    #         self._model = additional_model
    #     else:
    #         raise NotImplementedError("additional_model {} not recognised.".format(additional_model))

    #     # Initialise the george.GP object
    #     if additional_model is None or additional_model == 'y_offset':
    #         self.gp = george.GP(kernel=self._kernel,
    #                             white_noise=1.0,
    #                             mean=np.nanmedian(ts['f']),
    #                             fit_white_noise=True,
    #                             fit_mean=True)
    #     elif isinstance(additional_model, (int, float)):
    #         self.gp = george.GP(kernel=self._kernel,
    #                             white_noise=1.0,
    #                             mean=additional_model,
    #                             fit_whitetsn=True)
    #     else:
    #         self.gp = george.GP(kernel=self._kernel,
    #                             white_noise=1.0,
    #                             mean=self._model,
    #                             fit_white_noise=True,
    #                             fit_mean=True)

    #     # Model parameters and names
    #     if additional_model in ('y_offset', None) or isinstance(additional_model, (int, float)):
    #         self._model_parameters = ('y_offset',)
    #         self._model = None
    #     else:
    #         self._model_parameters = tuple(
    #             self._model.get_parameter_names(include_frozen=True))

    #     self._hp_names_local = self._model_parameters + \
    #                            self._local_parameters + \
    #                            self.LCKernel.parameter_names

    #     self._hp_names_gp = self.gp.parameter_names

    #     if np.shape(self._hp_names_local) != np.shape(self._hp_names_gp):
    #         raise ValueError("Internal and GP parameter shape doesn't match.")

    #     # White noise parameter bounds are separate (on 2ln_wn)
    #     self.wn_bounds = [-20.0, 0.0]

    #     # Set initial white noise
    #     if infer_wn:
    #         self.set_parameter(
    #             name='2ln_wn', local_name=True,
    #             value=lc_preparation.get_white_noise(ts, chunk_size=10))

    #     # Freeze mask doesn't need to be set; it's already in gp
    #     # Freeze the y_offset parameter in all default cases
    #     if additional_model in ('ramp', None) or isinstance(additional_model, (int, float)):
    #         self.freeze_parameter('y_offset')

    #     # Set-up the internal timeseries (detached from input)
    #     self.set_ts(ts)
    #     # Set up the initial basis (ib)
    #     # TODO: mask flares seems like it shouldn't be internal surely
    #     # TODO: replace with select basis
    #     ib = ts[~lc_preparation.mask_flares(ts)].copy()
    #     ib = ib.assign(opt_basis=False, gp_basis=False)
    #     opt_idx = np.random.choice(
    #         ib.index, N_opt if N_opt < len(ib) else len(ib), replace=False)
    #     gp_idx = np.random.choice(
    #         ib.index, N_opt if N_opt < len(ib) else len(ib), replace=False)
    #     ib.loc[opt_idx, 'opt_basis'] = True
    #     ib.loc[gp_idx, 'gp_basis'] = True

    #     self.set_basis(ib)

    #     # Set internal values
    #     self.set_ocut(ocut)
    #     self.__detrending_hp = None		# Used to cache the latest hp
    #                                     # that were used to detrend the
    #                                     # saved lightcurve. Done so that
    #                                     # we can check if it's detrended

    #     self.OversamplingPopulationError = OversamplingPopulationError
    #     self.NonPositiveDefiniteError = NonPositiveDefiniteError
    #     self.PriorInitialisationError = PriorInitialisationError

    #     X_0 = self.get_basis(only='gp_basis')[self._X_cols].values
    #     self.compute(X_0)

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
        """Decorator around gp.predict to handle Exceptions.

        NOTE: this does not split the timeseries into chunks. It will
        attempt to call it in one piece. Use predict_trend instead.
        """

        if self.N_gp is not None and self.N_gp < len(y_basis):
            warnings.warn("Using predict with {} points, while N_gp "
                          "is set to {}.".format(len(y_basis), self.N_gp))

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
            basis_subset
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

        # Separate the prediction into many chunks of 1000
        f_predict = np.empty(shape=len(X_predict), dtype=float)
        N_split = self.N_split

        for n_0 in N_split * np.arange(1+len(f_predict)//N_split):
            f_predict[n_0:n_0 + N_split], _ = self.predict(
                y_basis, X_predict[n_0:n_0 + N_split])

        return f_predict

    def get_spatial_component(self, X_predict=None, ts_basis=None, hp=None):
        """Spatially correlated component at points X_predict.

        Gives:
            f_star_xy = K_star . K^-1 . f
        where
            K_star is the vector of coveriances between X_predict
            and X_basis, and the K's here to the spatial kernel.

        Args:
            X_predict (pd.DataFrame): the points on which to predict
                the trend. Must contain columns ['x', 'y', 't'].
                Default: self._ts.
            ts_basis (pd.DataFrame): the points to use as the basis
                from which to regress the trend. Must contain columns
                ['f', 'x', 'y', 't'].
                Default: self._ts_basis.
            hp (1D array): vector for the GP hyperparameters.
                Default: self.hp

        Returns:
            f_predict (1D array): aka f_star_xy; the spatially
                correlated component in the lightcurve at points
                X_predict
        """

        if hp is not None:
            self.set_hp(hp)
        if ts_basis is None:
            ts_basis = self.get_basis(only='gp_basis')
        if X_predict is None:
            X_predict = self.get_ts()[self._X_cols].values
        elif isinstance(X_predict, pd.DataFrame):
            X_predict = X_predict[self._X_cols].values

        f_basis = ts_basis['f'].values
        X_basis = ts_basis[self._X_cols].values

        self.compute(X_basis)

        # Separate the prediction into many chunks of 1000
        f_predict = np.empty(shape=len(X_predict), dtype=float)
        N_split = self.N_split

        for n_0 in N_split * np.arange(1 + len(f_predict)//N_split):
            X_sub = X_predict[n_0:n_0 + N_split]
            K_star = self.gp.kernel.k1.get_value(x1=X_sub, x2=X_basis)
            f_sub = np.matmul(K_star, self.gp.apply_inverse(f_basis))
            f_predict[n_0:n_0 + N_split] = f_sub

        return f_predict

    def get_temporal_component(self, X_predict=None,
                               ts_basis=None, hp=None):
        """Temporally correlated component at points X_predict.

        Gives:
            f_star_xy = K_star . K^-1 . f
        where
            K_star is the vector of coveriances between X_predict
            and X_basis, and the K's here to the spatial kernel.

        Args:
            X_predict (pd.DataFrame): the points on which to predict
                the trend. Must contain columns ['x', 'y', 't'].
                Default: self._ts.
            ts_basis (pd.DataFrame): the points to use as the basis
                from which to regress the trend. Must contain columns
                ['f', 'x', 'y', 't'].
                Default: self._ts_basis.
            hp (1D array): vector for the GP hyperparameters.
                Default: self.hp

        Returns:
            f_predict (1D array): aka f_star_xy; the spatially
                correlated component in the lightcurve at points
                X_predict
        """

        if hp is not None:
            self.set_hp(hp)
        if ts_basis is None:
            ts_basis = self.get_basis(only='gp_basis')
        if X_predict is None:
            X_predict = self.get_ts()[self._X_cols].values
            # t_predict = self.get_ts()[self._model_cols].values
        elif isinstance(X_predict, pd.DataFrame):
            X_predict = X_predict[self._X_cols].values

        f_basis = ts_basis['f'].values
        X_basis = ts_basis[self._X_cols].values

        self.compute(X_basis)

        # Separate the prediction into many chunks of 1000
        f_predict = np.empty(shape=len(X_predict), dtype=float)
        N_split = self.N_split

        for n_0 in N_split * np.arange(1 + len(f_predict)//N_split):
            X_sub = X_predict[n_0:n_0 + N_split]
            K_star = self.gp.kernel.k2.get_value(x1=X_sub, x2=X_basis)
            f_sub = np.matmul(K_star, self.gp.apply_inverse(f_basis))
            f_predict[n_0:n_0 + N_split] = f_sub

        return f_predict

    def select_basis(self, N_opt=None, N_gp=None, ts=None,
                     cut_outliers=True, try_subset=True,
                     cut_transits=True, save=True, save_o_flag=True,
                     quiet=True, cut_flares=False, **mask_kwargs):
        """Removes outliers and samples an ts_basis from a lightcurve.

        TODO: get rid of ts_basis argument

        The ts_basis has an column called 'sub_flag', determined by N.
        The basis are the points that have sub_flag = True; this
        determines the subset of the basis which is used for
        optimization. When retrieving the basis, to ask for
        the full basis or the subset, use .get_basis(full=...)

        Args:
            N (int, optional): number of points to use.
            ts (pd.Dataframe, optional): uses self._ts by default
            cut_outliers (bool): if False, doesn't remove outliers.
            cut_transits (bool): if False, doesn't remove transit points,
                taken from the 't_flag' column in ts.
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

        if N_opt is not None and N_gp is not None:
            subsize = min(len(ts), 2*(N_opt+N_gp))
        else:
            # Can't try a subset if all the points are needed
            subsize = len(ts)
            try_subset = False

        # Do this first so it's not part of any subsets
        if cut_transits and 't_flag' in ts.columns:
            ts = ts[~ts.t_flag]

        if try_subset and cut_outliers:
            ts_basis = pd.DataFrame(columns=ts.columns)

            while (~ts_basis.o_flag).sum() < (N_opt + N_gp):
                try:
                    add_basis = ts.sample(subsize, replace=False)
                except ValueError:
                    # means we oversampled
                    # if quiet:
                    #     warnings.warn("basis set is undersampled. "
                    #                   "There were less non-outlier points "
                    #                   "than N_opt + N_gp.")
                    if quiet:
                        if len(ts) > 0:
                            add_basis = ts
                        else:
                            break
                    else:
                        raise OversamplingPopulationError

                # Take care not to allow replacement of points
                ts = ts[~ts.index.isin(add_basis.index)]
                outlier_mask = self.mask_outliers(ts=add_basis, **mask_kwargs)
                add_basis = add_basis[~outlier_mask]
                ts_basis = pd.concat(objs=[ts_basis, add_basis],
                                     ignore_index=True)
        elif cut_outliers:
            outlier_mask = self.mask_outliers(ts=ts, **mask_kwargs)
            ts_basis = ts[~outlier_mask]
        else:
            ts_basis = ts

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
        opt_idx = np.random.choice(ts_basis.index, N_opt, replace=False)
        gp_idx = np.random.choice(ts_basis.index, N_gp, replace=False)
        ts_basis.loc[opt_idx, 'opt_basis'] = True
        ts_basis.loc[gp_idx, 'gp_basis'] = True

        if save:
            self.set_basis(ts_basis)
        if not try_subset and save_o_flag and cut_outliers:
            self.set_ts(ts.assign(o_flag=outlier_mask))

        return ts_basis


    # Utilities and internals
    # -----------------------

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
        return None
        raise NotImplementedError

        # ts = ts if ts is not None else self._ts
        # if remove_outliers:
        #     ts = ts[~ts.o_flag]
        # if remove_transits and 't_flag' in ts.columns:
        #     ts = ts[~ts.t_flag]

        # if columns == 'all':
        #     columns = ('f', 'f_detrended')
        # elif isinstance(columns, str):
        #     columns = (columns,)

        # cdpps = dict()
        # for col in columns:
        #     cdpps[col] = lc_utils.calc_cdpp(ts, column=col)

        # if len(columns) == 1:
        #     return cdpps[columns[0]]
        # else:
        #     return cdpps



    # Aliases to allow more seamless integration with previous
    # --------------------------------------------------------
    # lcf = ts
    # lcf_basis = ts_basis
    # detrend_lightcurve = detrend

    # get_lcf = get_ts
    # set_lcf = set_ts
