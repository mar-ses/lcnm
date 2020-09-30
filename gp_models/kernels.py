"""Contains the definitions and utility functions for running a GP object.

NOTE on "CONVENTION":
    - pass around direct np.arrays, not lcf DataFrames;
        i.e X = lcf[['t', 'x', 'y']].values
            f = lcf.f.values
        The object here don't know how lcf objects work

NOTE on STRUCTURE: 
    - Aim for a structure where we separate Kernel and GPDetrender objects.
    - GPDetrender objects may even be general for a kernel
    - Kernel must contain the GP kernels, as well as DE bounds and priors.
    - GPDetrender object must be able to calculate the lnposterior
    - GPDetrender object must also take care of the full detrending
    - GPDetrender contains the data only through GP

NOTE POSSIBILITIES:
    - LCGP object also automatically

NOTE on short cadence extension:
    - Will create an SCGPDetrender (or SCDetrended) which will overload
      the necessary methods with the optimized short-cadence algorithms.

NOTE on paradigm:
    - TODO: go back and look at structure again. It is getting a little
      bloated and "general." Keep this in terms of very specific objects
      that to very specific jobs. I.e one is a prior/kernel (switch to
      prior perhaps). Another is a GP wrapper, with detrend methods,
      and lnlikelihood/_lnprior exposure; possibly two different
      kinds of optimize methods (DE and powell), and finally, tied
      to the detrending/predict methods; an outlier highlighter and
      parameter conversion.

      Everything else; i.e leading the detrending process, over 1
      stage or multiple, is done outside (outside this module even).
    - I would say Kernel object is still useful in a way, but whether
      it's kept or not, I think a prior object should be made.
    - Problem is prior depends on the kernel and parametrisation. So
      redo prior as a general thing that only calculates probabilities,
      but Kernel object sets the DE domain, and introduces a wrapper
      (in fact a decorator) that will convert from hp to prior params.
    - In fact, perhaps Prior should just be completely general,
      and instead its values (hyperprior) and dimension are defined
      by the Kernel object.
"""

import warnings
# from collections import OrderedDict

import numpy as np
from scipy.stats import truncnorm, norm

from george import kernels

# Two cases: "classic" and "quasi-periodic"
#classic_initial_hp = [-13.0, -12.86, -3.47, -4.34, -12.28, 2.32]
classic_initial_hp = [-13.0, -12.86, -3.47, -4.34, -12.28, 0.32]

# TODO: Do a full study of the distributions of hp posteriors for
#		different stellar populations, and redefine the priors in
#		terms of that, but in 'standard' or natural coordinates.

# NOTE on the GP hyperparameters for ExpSquared.
#
# george defines exponential squared kernels as:
#
#	k(r^2) = (A) * exp(-r^2 / 2)
#
#	where r^2 is defined under the metric C:
#	r^2 = (x_i - x_j)^T . C^-1 . (x_i - x_j)	where i,j don't index
#												the vector components
#												but the individual vectors
#												themselves (or observations)
#
# When defining a two dimensional ExpSquaredKernel as:
#
#	kernel = 1 * george.ExpSquaredKernel(metric=[2, 3], ndim=2)
#
# the george vector is np.ln([2,3]), and the metric C is:
#
#	C = (2	0)
#		(0	3)
#
# thus, C is the matrix of covariances (IN THE KERNEL):
#
#	k(r^2) = 1 * exp(-r^2 / 2)
#
#	where r^2 = (x_i - x_j)^2 / 2*C_11 + (x)
#
# SUMMARY:
#
# george vector components for the var/eta are:
#
#	hp_var = ln(var) = ln(1 / (2eta)) = -ln(2) - ln(eta)

# Structure, to be extended to all
# --------------------------------
# To be the same as the model_objects, but no frozen mask or values
#
# Attributes:
# parameter_names, default_values, bounds (array), _hpp
#
#
#

class LCKernel(object):
    """Base kernel class for lcnm detrending objects.

    NOTE: base interface doesn't know number of X dimensions.

    NOTE: .kernel object doesn't have bounds implemented properly.
    Therefore, the bounds in here are only "suggested." It is expected
    that the detrender object will respect them, and they define the
    prior boundaries as well.
    """

    def __init__(self, kernel, parameter_names, default_values,
                 keyword=None, bound_dict=None):
        """Performs the standard initialisation routines.

        NOTE: doesn't set the kernel. The kernel needs to be created
        after the bounds are set, which means in the child object's
        __init__ function, after the bounds and names have been
        initialised.
        """

        #self.set_hyperpriors()
        #self.set_bounds()
        self.parameter_names = tuple(parameter_names)
        self.default_values = default_values
        self.kernel = kernel
        self.kernel.set_parameter_vector(np.array(self.default_values))

        self.set_hyperpriors(keyword=keyword)
        self.set_bounds(keyword=keyword, bound_dict=bound_dict)

        # Additional potential tools
        self.get_parameter_vector = self.kernel.get_parameter_vector
        self.set_parameter_vector = self.kernel.set_parameter_vector

    # Basic propers (not so important)

    def __len__(self):
        """Total number of active parameters."""
        return len(self.parameter_names)

    @property
    def full_size(self):
        """The total number of parameters (include inactive)."""
        return len(self)

    def get_parameter_names(self):
        return self.parameter_names

    # Bounds
    # ------
    # bounds is currently a problem. kernel.parameter_bounds belongs to
    # to the kernel object and seemingly cannot be modified after it's
    # been created. One cannot change specific elements, for example:
    # self.kernel.parameter_bounds[0][1] = 0 fails
    # Therefore, for my prior, I will use a different property,
    # self.bounds. The prior must only interact with this.
    #
    # Check back on: https://github.com/dfm/george/pull/114
    #
    # If george implements modifiable bounds, then change it so that
    # we only refer to self.kernel.parameter_bounds

    def get_bound_dict(self):
        """Returns the LCKernel.bounds matched to parameter names.

        NOTE: not george.kernels.parameter_bounds
        """
        return dict(zip(self.parameter_names, self.bounds))

    # TODO: find and fix get_parameter_bounds
    def get_bounds(self):
        """Returns the LCKernel.bounds.

        NOTE: not george.kernels.parameter_bounds
        """
        return self.bounds

    # This should be overridden by each child object with its own
    # specific way of choosing default bounds.
    def set_bounds(self, keyword=None, bound_dict=None):
        """Sets bounds on the parameters."""

        warnings.warn("Using LCKernel.set_bounds. This should have "
                      "been overridden by a child object. The bounds "
                      "will be plus-minus infinity (unbounded). This "
                      "will break things if the parameters are in log "
                      "form.")

        self.bounds = [[-np.inf, np.inf]]*len(self)

    def log_prior(self, vector):
        """Compute the log_prior of the current parameters.

        NOTE: the parameter order absolutely matters, must be the same
        as the order in parameter_names and default_values.
        """
        raise NotImplementedError

        #lp_value = 0.0
        #for i in range(len(self)):
        #	lp_value += self.hpp[self.parameter_names[i]](vector[i])
        #return lp_value

    def grad_log_prior(self, vector):
        """Compute the log_prior of the current parameters."""
        raise NotImplementedError

        # grad = 0.0
        # for i in range(len(self)):
        # 	grad += self.hpp_grad[self.parameter_names[i]](vector[i])
        # return grad

    def set_hyperpriors(self, keyword=None):
        """Creates a dict of functions as the hpp for each parameter."""
        raise NotImplementedError


# K2 kernels
# ----------

class ClassicK2Kernel(LCKernel):
    """Standard K2 kernel for temporal and spatial noise correlation.
    """

    def __init__(self, keyword=None, **kwargs):
        """Initialises the priors, initial hp values, and kernel."""

        # Set up kernel
        # -------------
        k_spatial = 1.0 * kernels.ExpSquaredKernel(
            metric=[1.0, 1.0], ndim=3, axes=[0, 1])
        k_temporal = 1.0 * kernels.ExpSquaredKernel(
            metric=1.0, ndim=3, axes=2)
        k_total = k_spatial + k_temporal

        if keyword in ('long', 'long_timescale'):
            default_sigt = 16
        elif isinstance(keyword, (int, float)):
            default_sigt = keyword + 1e-5
        else:
            default_sigt = np.exp(0.36/2)

        super().__init__(kernel=k_total,
                         parameter_names=('ln_Axy', '2ln_sigx', '2ln_sigy',
                                          'ln_At', '2ln_sigt'),
                         default_values=(-12.86, -3.47, -4.34,
                                         -12.28, 2*np.log(default_sigt)),
                         keyword=keyword,
                         **kwargs)

        self.default_X_cols = ['x', 'y', 't']

    def log_prior(self, vector=None):
        """Calculate the log_prior for a given parameter vector.

        Args:
            vector (np.array): if None, will take the
                parameter_vector in self.kernel, which should be
                the same as in the GP object.
                NOTE: expects the *full* vector
        """

        if vector is None:
            vector = self.kernel.get_parameter_vector(include_frozen=True)

        for i, val in enumerate(vector):
            if val < self.bounds[i][0] or val > self.bounds[i][1]:
                return -np.inf

        logpdf_2ln_sigx = norm.logpdf(vector[1],
                                      loc=self._hpp['2ln_sigx'][0],
                                      scale=self._hpp['2ln_sigx'][1])

        logpdf_2ln_sigy = norm.logpdf(vector[2],
                                      loc=self._hpp['2ln_sigy'][0],
                                      scale=self._hpp['2ln_sigy'][1])

        logpdf_2ln_sigt = norm.logpdf(vector[4],
                                      loc=self._hpp['2ln_sigt'][0],
                                      scale=self._hpp['2ln_sigt'][1])

        return logpdf_2ln_sigx + logpdf_2ln_sigy + logpdf_2ln_sigt

    def set_bounds(self, keyword=None, bound_dict=None):
        """Bounds are in *actual* scale; not transformed.

        Args:
            keyword: sets different bounds.
                - high_frequency: lower bound of 0.1 for periods down
                  to 0.1 days
        """

        if keyword is None or keyword in ('standard', 'main'):
            min_sigt = 0.5
        elif keyword in ('long', 'long_timescale'):
            min_sigt = 15
        elif isinstance(keyword, (int, float)):
            min_sigt = keyword

        # order Axy, sigx, sigy, At, sigt
        self.bounds = np.array([[-20.0, 2.5],
                                [2*np.log(1e-7), 2*np.log(70)],
                                [2*np.log(1e-7), 2*np.log(70)],
                                [-20.0, 2.5],
                                [2*np.log(min_sigt), 2*np.log(100)]])

        # bound_dict overwrites anything else
        if bound_dict is not None:
            for key in bound_dict:
                idx = self.parameter_names.index(key)
                self.bounds[idx] = list(bound_dict[key])

    def set_hyperpriors(self, keyword=None):
        self._hpp = dict()

        # TODO: the priors for sigx and sigt have drastically changed
        # Mean and std of 2ln_sigx
        self._hpp['2ln_sigx'] = [np.log(17), np.log(8.0)]

        # Mean and std of 2ln_sigy
        self._hpp['2ln_sigy'] = [np.log(17), np.log(8.0)]

        # Mean and std of 2ln_sigt
        self._hpp['2ln_sigt'] = [2*np.log(20), 3.0]


class QuasiPeriodicK2Kernel(LCKernel):
    """Quasi-periodic kernel for temporal and spatial noise correlation.
    """

    def __init__(self, P=None, *args, **kwargs):
        """Initialises the priors, initial hp values, and kernel."""

        # Set up kernel
        # -------------
        k_spatial = 1.0 * kernels.ExpSquaredKernel(
                                            metric=[1.0, 1.0],
                                            ndim=3, axes=[0, 1])
        k_temporal = 1.0 * kernels.ExpSquaredKernel(
                                            metric=1.0,
                                            ndim=3, axes=2) \
                         * kernels.ExpSine2Kernel(
                                             gamma=2, log_period=1,
                                            ndim=3, axes=2)
        k_total = k_spatial + k_temporal

        if P is None:
            P = 0.5

        # NOTE: sigt always starts as multiple of the period
        super().__init__(kernel=k_total,
                         parameter_names=('ln_Axy', '2ln_sigx', '2ln_sigy',
                                          'ln_At', '2ln_sigt', 'gamma', 'lnP'),
                         default_values=(-12.86, -3.47, -4.34,
                                         -12.28, 2*np.log(4*P),
                                         1.0, np.log(P)),
                         *args, **kwargs)

        # self.set_hyperpriors(keyword=keyword)
        # self.set_bounds(keyword=keyword)
        # self.parameter_names = ('ln_Axy', '2ln_sigx', '2ln_sigy',
        # 				 		'ln_At', '2ln_sigt', 'gamma', 'lnP')
        # self.default_values = (-12.86, -3.47, -4.34, -12.28,
        # 					   max(2*np.log(4*P), self.bounds[4][0] + 1e-6),
        # 					   1.0, np.log(P))
        # self.kernel = k_total
        # self.kernel.set_parameter_vector(np.array(self.default_values))

        # # Additional potential tools
        # self.get_parameter_vector = self.kernel.get_parameter_vector
        # self.set_parameter_vector = self.kernel.set_parameter_vector

        if np.log(P) < self.bounds[-1][0] or np.log(P) > self.bounds[-1][1]:
            raise PriorInitialisationError((
                "Initial period is out of bounds\nperiod: {},\n"
                "lnP: {}, \nbounds: {}".format(P, np.log(P), self.bounds[-1])))
        elif not np.isfinite(self.log_prior(self.default_values)):
            raise PriorInitialisationError((
                "Initial hyperparameter values are out of "
                "prior bounds.\n"
                "hp_default: {}\n"
                "bounds: {}\n"
                "P: {}".format(self.default_values, self.bounds, P)))

        self.default_X_cols = ['x', 'y', 't']

    def log_prior(self, vector=None):
        """Calculate the log_prior for a given parameter vector.

        Args:
            vector (np.array): if None, will take the
                parameter_vector in self.kernel, which should be
                the same as in the GP object.
                NOTE: expects the *full* vector
        """

        if vector is None:
            vector = self.kernel.get_parameter_vector(include_frozen=True)

        for i, val in enumerate(vector):
            if val < self.bounds[i][0] or val > self.bounds[i][1]:
                return -np.inf

        # Period must be less than temporal scale
        # ln(2) = 0.69314718055994529
        if vector[-1] > vector[4]/2:
            return - np.inf

        logpdf_2ln_sigx = norm.logpdf(vector[1],
                                      loc=self._hpp['2ln_sigx'][0],
                                      scale=self._hpp['2ln_sigx'][1])

        logpdf_2ln_sigy = norm.logpdf(vector[2],
                                      loc=self._hpp['2ln_sigy'][0],
                                      scale=self._hpp['2ln_sigy'][1])

        logpdf_2ln_sigt = norm.logpdf(vector[4],
                                      loc=self._hpp['2ln_sigt'][0],
                                      scale=self._hpp['2ln_sigt'][1])

        return logpdf_2ln_sigx + logpdf_2ln_sigy + logpdf_2ln_sigt

    def set_bounds(self, keyword=None, bound_dict=None):
        """Bounds are in *actual* scale; not transformed.
        
        Args:
            keyword: sets different bounds.
                - high_frequency: lower bound of 0.1 for periods down
                  to 0.1 days
                -
        """

        P_lower, P_upper = 0.5, 40
        sigt_lower = 1.0

        if keyword in ('hf', 'high_frequency'):
            P_lower = 0.1
            sigt_lower = 0.2
        elif keyword in ('uhf', 'ultra_high_frequency'):
            P_lower = 0.02
            sigt_lower = 0.04
        else:
            print("No keyword:", keyword)

        # order At, sigt, g, lnP
        self.bounds = np.array([[-20.0, 2.5],
                                [2*np.log(1e-7), 2*np.log(70)],
                                [2*np.log(1e-7), 2*np.log(70)],
                                [-20.0, 2.5],
                                [2*np.log(sigt_lower), 2*np.log(300)],
                                [0, 20],
                                [np.log(P_lower), np.log(P_upper)]])

        # bound_dict overwrites anything else
        if bound_dict is not None:
            for key in bound_dict:
                idx = self.parameter_names.index(key)
                self.bounds[idx] = list(bound_dict[key])

    def set_hyperpriors(self, keyword=None):
        self._hpp = dict()

        # TODO: the priors for sigx and sigt have drastically changed
        # Mean and std of 2ln_sigx
        self._hpp['2ln_sigx'] = [np.log(17), np.log(8.0)]

        # Mean and std of 2ln_sigy
        self._hpp['2ln_sigy'] = [np.log(17), np.log(8.0)]

        # Mean and std of 2ln_sigt
        self._hpp['2ln_sigt'] = [2*np.log(40), 3.0]

    def grad_log_prior(self, vector):
        """Compute the log_prior of the current parameters."""
        raise NotImplementedError


# Special kernels
# ---------------

class QuasiPeriodic1DKernel(LCKernel):

    #_param_names = ('wn', 'At', 'sigt', 'g', 'lnP')
    #_default_hp = (-13.0, -12.28, 0.36, 1.0, -1.0)
    #_kernel_type = 'quasiperiodic-1D'

    def __init__(self, P=None, **kwargs):
        """Initialises the priors, initial hp values, and kernel."""

        # Set up kernel
        k_temporal = 1.0 * kernels.ExpSquaredKernel(metric=1.0, ndim=1) \
            * kernels.ExpSine2Kernel(gamma=2, log_period=1, ndim=1)

        if P is None:
            P = 0.5

        # NOTE: sigt always starts as multiple of the period
        super().__init__(
            kernel=k_temporal,
            parameter_names=('ln_At', '2ln_sigt', 'g', 'lnP'),
            default_values=(-12.28, 2*np.log(4*P), 1.0, np.log(P)),
            **kwargs)

        if not np.isfinite(self.log_prior(self.default_values)):
            raise PriorInitialisationError(
                "Initial hyperparameter values are out of prior bounds.")

        self.default_X_cols = 't'

    def log_prior(self, vector=None):
        """Calculate the log_prior for a given parameter vector.

        Args:
            vector (np.array): if None, will take the
                parameter_vector in self.kernel, which should be
                the same as in the GP object.
                NOTE: expects the *full* vector
        """

        if vector is None:
            vector = self.kernel.get_parameter_vector(include_frozen=True)

        for i, val in enumerate(vector):
            if val < self.bounds[i][0] or val > self.bounds[i][1]:
                return -np.inf

        # Period must be less than temporal scale
        # ln(2) = 0.69314718055994529
        if vector[-1] > vector[1]/2:
            return - np.inf

        logpdf_2ln_sigt = norm.logpdf(vector[1],
                                      loc=self._hpp['2ln_sigt'][0],
                                      scale=self._hpp['2ln_sigt'][1])

        return logpdf_2ln_sigt

    def set_bounds(self, keyword=None, bound_dict=None):
        """Bounds are in *actual* scale; not transformed.

        Args:
            keyword: sets different bounds.
                - high_frequency: lower bound of 0.1 for periods down
                  to 0.1 days
        """

        P_lower, P_upper = 0.5, 40
        sigt_l, sigt_u = 1.0, 300

        if keyword in ('hf', 'high_frequency'):
            P_lower = 0.1
            sigt_l = 0.4
        elif keyword in ('uhf', 'ultra-high-frequency', 'brown-dwarf'):
            P_lower = 0.01
            sigt_l = 0.04

        # order At, sigt, g, lnP
        self.bounds = np.array([[-20.0, 2.5],
                                [2*np.log(sigt_l), 2*np.log(sigt_u)],
                                [0, 20],
                                [np.log(P_lower), np.log(P_upper)]])



        # bound_dict overwrites anything else
        if bound_dict is not None:
            for key in bound_dict:
                idx = self.parameter_names.index(key)
                self.bounds[idx] = list(bound_dict[key])


    def set_hyperpriors(self, keyword=None):
        self._hpp = dict()

        # Mean and std of 2ln_sigt
        self._hpp['2ln_sigt'] = [2*np.log(20), 3.0]


        # Previous definitions in dictionary form

        # def At_prior(value):
        # 	"""Bounded uniform prior."""
        # 	if value < self.bounds['At'][0] or value > self.bounds['At'][1]:
        # 		return -np.inf
        # 	else:
        # 		return 0.0

        # def sigt_prior(value):
        # 	return truncnorm.logpdf(-np.log(2) - value,
        # 							*rescale_trunclims((-12.7, 2.5,
        # 												-1.0, 2.25)))

        # def g_prior(value):
        # 	if value < self.bounds['g'][0] or value > self.bounds['g'][1]:
        # 		return - np.inf
        # 	else:
        # 		return 0.0

        # def P_prior(value):
        # 	if value < self.bounds['P'][0] or value > self.bounds['P'][1]:
        # 		return - np.inf
        # 	elif hp[6] > (0.69314718055994529 + hp[4]/2):		
        # 		# Block the decay timescale from being shorter than
        # 		# the period (or half period)
        # 		# ln(2) = 0.69314718055994529
        # 		return - np.inf

    def grad_log_prior(self, vector):
        """Compute the log_prior of the current parameters."""
        raise NotImplementedError


# Matern Kernels
# --------------

class Matern52K2Kernel(LCKernel):
    """Matern kernel.

    Quick and dirty for testing purposes.
    """

    def __init__(self, keyword=None, *args, **kwargs):
        """Initialises the priors, initial hp values, and kernel."""

        # Set up kernel
        # -------------
        k_spatial = 1.0 * kernels.Matern52Kernel(
            metric=[1.0, 1.0], ndim=3, axes=[0, 1])
        k_temporal = 1.0 * kernels.Matern52Kernel(
            metric=1.0, ndim=3, axes=2)
        k_total = k_spatial + k_temporal

        if keyword in ('long', 'long_timescale'):
            default_sigt = 16
        elif isinstance(keyword, (int, float)):
            default_sigt = keyword + 1e-5
        else:
            default_sigt = np.exp(0.36/2)

        super().__init__(kernel=k_total,
                         parameter_names=('ln_Axy', '2ln_sigx', '2ln_sigy',
                                          'ln_At', '2ln_sigt'),
                         default_values=(-12.86, -3.47, -4.34,
                                         -12.28, 2*np.log(default_sigt)),
                         keyword=keyword,
                         *args, **kwargs)

        self.default_X_cols = ['x', 'y', 't']

    def log_prior(self, vector=None):
        """Calculate the log_prior for a given parameter vector.

        Args:
            vector (np.array): if None, will take the
                parameter_vector in self.kernel, which should be
                the same as in the GP object.
                NOTE: expects the *full* vector
        """

        if vector is None:
            vector = self.kernel.get_parameter_vector(include_frozen=True)

        for i, val in enumerate(vector):
            if val < self.bounds[i][0] or val > self.bounds[i][1]:
                return -np.inf

        logpdf_2ln_sigx = norm.logpdf(vector[1],
                                      loc=self._hpp['2ln_sigx'][0],
                                      scale=self._hpp['2ln_sigx'][1])

        logpdf_2ln_sigy = norm.logpdf(vector[2],
                                      loc=self._hpp['2ln_sigy'][0],
                                      scale=self._hpp['2ln_sigy'][1])

        logpdf_2ln_sigt = norm.logpdf(vector[4],
                                      loc=self._hpp['2ln_sigt'][0],
                                      scale=self._hpp['2ln_sigt'][1])

        return logpdf_2ln_sigx + logpdf_2ln_sigy + logpdf_2ln_sigt

    def set_bounds(self, keyword=None, bound_dict=None):
        """Bounds are in *actual* scale; not transformed.

        Args:
            keyword: sets different bounds.
                - high_frequency: lower bound of 0.1 for periods down
                  to 0.1 days
        """

        if keyword is None or keyword in ('standard', 'main'):
            min_sigt = 0.5
        elif keyword in ('long', 'long_timescale'):
            min_sigt = 15
        elif isinstance(keyword, (int, float)):
            min_sigt = keyword

        # order Axy, sigx, sigy, At, sigt
        self.bounds = np.array([[-20.0, 2.5],
                                [2*np.log(1e-7), 2*np.log(70)],
                                [2*np.log(1e-7), 2*np.log(70)],
                                [-20.0, 2.5],
                                [2*np.log(min_sigt), 2*np.log(100)]])

        # bound_dict overwrites anything else
        if bound_dict is not None:
            for key in bound_dict:
                idx = self.parameter_names.index(key)
                self.bounds[idx] = list(bound_dict[key])

    def set_hyperpriors(self, keyword=None):
        self._hpp = dict()

        # Mean and std of 2ln_sigx
        self._hpp['2ln_sigx'] = [np.log(17), np.log(8.0)]

        # Mean and std of 2ln_sigy
        self._hpp['2ln_sigy'] = [np.log(17), np.log(8.0)]

        # Mean and std of 2ln_sigt
        self._hpp['2ln_sigt'] = [2*np.log(20), 3.0]


# --------------------------
# Utility and work functions
# --------------------------

def rescale_trunclims(p, return_full=False):
    """Gives the scaled limits that must be entered into trunknorm.

    Args:
        p (array-like): (lower, upper, mean, std)
        return_full (bool): if True, returns all four parameters
            so it can be used as an *arg

    Returns:
        a, b (the truncnorm limits)
            + mean, std (if return_full)
    """

    if return_full:
        return (p[0] - p[2]) / p[3], (p[1] - p[2]) / p[3], p[2], p[3]
    else:
        return (p[0] - p[2]) / p[3], (p[1] - p[2]) / p[3]


class PriorInitialisationError(ValueError):
    """Thrown when the prior verification fail."""

    def __init__(self, message=None, **kwargs):

        if message is None:
            message = ''

        if len(kwargs) > 0:
            message += "\nParsing:\n"
            for kw, arg in kwargs.items():
                message += '{}:\t\t{}\n'.format(kw, arg)
                setattr(self, kw, arg)

        super().__init__(message)
