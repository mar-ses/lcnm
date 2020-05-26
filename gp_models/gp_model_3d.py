"""Lightcurve model extended for 3 correlated parameters."""

import warnings
import numpy as np
import pandas as pd

from . import gp_model_base, model_objects

# ------------------
# 3D GP model object
# ------------------

class LCNoiseModel3D(gp_model_base.LCNoiseModel1D):
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

        K_star = self.gp.kernel.k1.get_value(x1=X_predict, x2=X_basis)
        f_predict = np.matmul(K_star, self.gp.apply_inverse(f_basis))

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
        elif isinstance(X_predict, pd.DataFrame):
            X_predict = X_predict[self._X_cols].values

        f_basis = ts_basis['f'].values
        X_basis = ts_basis[self._X_cols].values

        self.compute(X_basis)

        K_star = self.gp.kernel.k2.get_value(x1=X_predict, x2=X_basis)
        f_predict = np.matmul(K_star, self.gp.apply_inverse(f_basis))

        return f_predict

    def detrend(self, ts_predict=None, hp=None, save=True):
        """Produces a detrended timeseries.

        NOTE: uses 'gp_basis' by default.

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

        if ts_predict is None:
            ts_predict = self.ts.copy()
        if hp is not None:
            # Need to set the hp so that self._model works
            self.hp = hp

        if self._model is not None:
            f_model = self._model.get_value(ts_predict[self._model_cols])
        else:
            f_model = self.get_parameter('y_offset')

        f_spatial = self.get_spatial_component(ts_predict)
        f_temporal = self.get_temporal_component(ts_predict)

        # will not add the spatial + temporal in the detrended product
        # y_trend = self.predict_trend(ts_predict, subset_basis=subset_basis)
        # ts_predict['f_trend_gp'] = y_trend - f_model
        f_detrended = ts_predict['f'] - f_spatial - f_temporal

        # Special behaviour: ramp is supposed to be removed (detrender),
        # while other models aren't. So if the model is ramp, then
        # remove from detrended, otherwise keep in detrended.
        # i.e. in all cases, keep the y_offset in detrended.
        if isinstance(self._model, model_objects.RampModel):
            f_detrended += self.get_parameter('y_offset') - f_model

        ts_predict['f_model'] = f_model
        ts_predict['f_spatial'] = f_spatial
        ts_predict['f_temporal'] = f_temporal
        ts_predict['f_detrended'] = f_detrended

        if save:
            self.ts = ts_predict
            self.__detrending_hp = self.hp

        if 'opt_basis' in ts_predict or 'gp_basis' in ts_predict:
            warnings.warn('opt_basis and gp_basis in ts_predict.')

        return ts_predict #.drop(['opt_basis', 'gp_basis'])

    detrend_lightcurve = detrend
