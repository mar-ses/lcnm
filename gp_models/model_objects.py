"""Defines the ramp Model object to fit with GP."""

import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy import stats
#from george.modeling import Model

default_alpha = 0.2

# TODO: change t -> X (with X as the three dimensional vector)
# X is in shape N_samples*N_dim: [[x1, y1, t1], [x2, y2, t2], [...]]

# Final naming convention (and interaction with frozen/unfrozen):
# probably should end up the same as DFM's george scheme
#
# - parameter_names     : [tuple of str]
#						  by default, frozen included
#						  not a property, the actual array is stored
# - parameter_vector	: [np.array]
#					   	  by default, only unfrozen
# 						  alias - occasionally hp
# - unfrozen_mask		: [np.array, bool]
#						  True for active parameters
# - _bounds				: [np.array]
# 						  the minimum and maximum values allowed
#						  will be used directly by gradient-based samplers
#						  also may be used for truncated priors **SHOULD#
#
# In general, attributes starting with _... are actual attributes,
# for example _bounds. They are internal. Others are properties,
# meant to be used externally.
#
# NOTE:
# since these models are meant to be used with GP objects, they need
# to be capable and ready to take multi-dimensional X.
# The enforced convention is that the modelling axis (t) is in the
# last column of X.

class BaseModel(object):
	"""Defines a base model object.

	Must define parameter setting and getting, frozen parameters,
	names, as well as a basic idea of priors and of bounds too.
	"""

	def __init__(self, x_ndim=1, **kwargs):
		"""Defines the parameter name and value arrays, unfrozen
		masks and so on.

		NOTE: in inherited objects, must set bounds AFTER initiating
		the parent.
		
		Args:
			**kwargs: parameter names and values
		"""

		self.parameter_names = tuple(kwargs.keys())
		self.parameter_vector = np.array(list(kwargs.values()))
		self.unfrozen_mask = np.ones_like(self.parameter_vector, dtype=bool)
		
		if not hasattr(self, '_bounds'):
			self._bounds = np.array([-np.inf, np.inf]*self.full_size)

		# TODO: this is obsolete
		self._x_ndim = x_ndim

	def __len__(self):
		"""Total number of active parameters."""
		return np.sum(self.unfrozen_mask)

	@property
	def full_size(self):
		"""The total number of parameters (include inactive)."""
		return len(self.parameter_names)

	@property
	def vector_size(self):
		"""Alias for len()."""
		return len(self)

	def get_parameter_names(self, include_frozen=False):
		if not include_frozen:
			return tuple(n for n, f in zip(self.parameter_names, self.unfrozen_mask) if f)
		else:
			return self.parameter_names

	def get_parameter_vector(self, include_frozen=False):
		if not include_frozen:
			return self.parameter_vector[self.unfrozen_mask]
		else:
			return self.parameter_vector

	def set_parameter_vector(self, vector, include_frozen=False):
		if not include_frozen:
			self.parameter_vector[self.unfrozen_mask] = vector
		else:
			self.parameter_vector = vector

	def get_parameter_bounds(self):
	 	return self._bounds

	def get_parameter_dict(self, include_frozen=False):
		"""Returns a dict of parameters names:values."""

		return OrderedDict(zip(
			self.get_parameter_names(include_frozen=include_frozen),
			self.get_parameter_vector(include_frozen=include_frozen)
		))

	def get_parameter(self, name):
		return self.parameter_vector[self.parameter_names.index(name)]

	def set_parameter(self, name, value):
		self.parameter_vector[self.parameter_names.index(name)] = value

	def get_value(self):
		"""Returns the value of the model."""
		raise NotImplementedError("Only an interface placeholder.")

	def compute_gradient(self):
		"""Returns the gradient of the model in ALL parameters."""
		raise NotImplementedError("Only an interface placeholder.")

	def get_gradient(self, include_frozen=False, *args, **kwargs):
		"""Returns the gradient of the model.

		Args:
			t (array of floats): values of to calculate model at
			include_frozen (bool): if True, returns full_size
				gradient.
		"""

		if not include_frozen:
			return self.compute_gradient(*args, **kwargs)[self.unfrozen_mask]
		else:
			return self.compute_gradient(*args, **kwargs)

	def freeze_all_parameters(self):
		self.unfrozen_mask[:] = False

	def thaw_all_parameters(self):
		self.unfrozen_mask[:] = True

	def freeze_parameter(self, name):
		self.unfrozen_mask[np.array(self.parameter_names) == name] = False

	def thaw_parameter(self, name):
		self.unfrozen_mask[np.array(self.parameter_names) == name] = True

	def log_prior(self):
		"""Compute the log_prior of the current parameters."""
		raise NotImplementedError("Only an interface placeholder.")

	def grad_log_prior(self):
		"""Compute the log_prior of the current parameters."""
		raise NotImplementedError("Only an interface placeholder.")

	# def check_vector(self, vector):
	# 	for i, (a,b) in enumerate(self.get_bounds()):
	# 		v = vector[i]
	# 		# TODO: STUCK HERE

	
# Specific ramp model
# -------------------

class RampModel(BaseModel):
	"""Defines the ramp model object.

	y = A * exp(-t/alpha) + y_offset

	Internally, the values are not in log form. Interactions
	are NOT in log-form.

	To get the active parameters, use .parameter_names or
	.get_parameter_names; to get the full list of parameters,
	use .parameter_names.

	A, alpha, y_offset
	"""

	# TODO: obsolete, this should be in bounds
	_minA = 1e-8
	_maxA = 20.0

	def __init__(self, lcf=None, A_0=None, alpha_0=default_alpha,
				 y_offset=1.0, t_0=None, x_ndim=1):
		"""
		Args:
			lcf (pd.DataFrame, optional): used for estimating A
			A_0 (float, optional):
			alpha_0 (float, optional): default set at 0.2d
			y_offset (float, optional): the normalisation of the
				lightcurve in the y axis, by default frozen
			t_0 (float, optional): earliest time where ramp starts,
				for calculating initial values
			X_ndim (int): number of dimensions in the timeseries
				or general x component of the data being calculated on.
				If detrending from the general x, y, t GP, then need 3.
		"""

		if A_0 is not None:
			pass
		elif lcf is not None:
			A_0 = min(estimate_A(lcf.t, lcf.f), self._maxA)
		else:
			A_0 = 0.0
		# if abs(A_0) < self._minA:
		# 	A_0 = (A_0 > 0) * self._minA if A_0 != 0 else self._minA

		if t_0 is not None:
			self.t_0 = t_0
		elif t_0 is None and lcf is not None:
			self.t_0 = min(lcf.t)
		else:
			self.t_0 = 0.0

		# Bounds on A must not allow it to go to 0, since a log
		# needs to be taken in the prior and elsewhere.
		self._bounds = [[1e-8, 20.0], [0, 10], [-np.inf, np.inf]]

		super().__init__(A=A_0, alpha=alpha_0, y_offset=y_offset,
						 x_ndim=x_ndim)

	def get_value(self, x, quiet=False):
		"""Returns the value of the model.

		TODO: if _x_ndim = 1, will x still be a 2D array with 1 column,
			or will it be a fully 1D array?

		Args:
			x (array of floats): the values at which to calculate
				the model. If x_ndim > 1, will use the last column,
				x[:, -1], otherwise assumes it's a one dimensional
				array.
		"""

		if np.ndim(x) == 2:
			t = x[:, -1]
		else:
			t = x

		A, alpha, y_offset = self.parameter_vector
		f = A*np.exp(-(t - self.t_0)/alpha) + y_offset

		if not quiet and np.any(np.isnan(f)):
			raise ValueError("Possible exp overflow encountered in ramp.\n"
							 "Parameters: {}\n"
							 "Minimum t: {}\n"
							 "Nan count: {}".format(self.parameter_vector,
													min(t),
													np.sum(np.isnan(f))))
		return f

	def compute_gradient(self, x):
		"""Returns the gradient of the model in ALL parameters.

		Args:
			x (array of floats): the values at which to calculate
				the model. Only x[:,2] is assumed to be t, time.
		"""

		if self._x_ndim > 1:
			t = x[:, -1]
		else:
			t = x

		A, alpha, _ = self.parameter_vector
		value = self.get_value(x)
		gradient = np.array([
			value / A,
			value * (t - self.t_0)/alpha**2,
			np.ones_like(value, dtype=float)
		])

		# For lnA:
		# gradient = np.array([
		# 	value / (lnA*np.exp(lnA)),
		# 	value * (X[:, 2] - self.t_0)/alpha**2,
		# 	np.ones_like(value, dtype=float)
		# ])

		return gradient

	def get_gradient(self, x, include_frozen=False):
		"""Returns the gradient of the model.

		Args:
			x (array of floats): values of to calculate model at
			include_frozen (bool): if True, returns full_size
				gradient.
		"""

		if not include_frozen:
			return self.compute_gradient(x)[self.unfrozen_mask]
		else:
			return self.compute_gradient(x)

	def log_prior(self):
		"""Compute the log_prior of the current parameters.

		Currently standardised form; only preventing increasing
		exponentials.

		TODO: base it on bounds
		"""

		alpha = self.get_parameter('alpha')
		A = self.get_parameter('A')

		if alpha < self._bounds[1][0] or alpha > self._bounds[1][1]:
			return - np.inf

		return 0.5*stats.reciprocal.logpdf(max(abs(A), self._bounds[0][0]),
										   self._bounds[0][0],
										   self._bounds[0][1])

	def grad_log_prior(self):
		"""Compute the log_prior of the current parameters.

		Currently standardised form; only preventing increasing
		exponentials.

		TODO: base it on bounds
		"""

		raise NotImplementedError

		#if self.get_parameter('alpha') < 0:
		#	return - np.inf

		#return 0.0

	# def check_vector(self, vector):
	# 	for i, (a,b) in enumerate(self.get_bounds()):
	# 		v = vector[i]
	# 		# TODO: STUCK HERE

# Help functions

def estimate_initial(t, f):
	"""Estimates the initial A and t_0 from t and f.

	Assumes an initial alpha of 0.2.

	Returns [A, t_0]
	"""

	t0 = min(t)

	if isinstance(f, pd.Series):
		f = f.values
	diff = np.nanmean(f[:5])

	return [diff, t0]

def estimate_A(t, f):
	"""Estimates the initial A from t and f.

	Assumes an initial alpha of 0.2 and t_0 of min(lcf.t).
	"""

	t0 = min(t)

	if isinstance(f, pd.Series):
		f = f.values
	diff = np.nanmean(f[:5]) - np.nanmedian(f)

	return diff

def estimate_lnA(t, f):
	"""Estimates the initial A from t and f.

	Assumes an initial alpha of 0.2.
	"""

	t0 = min(t)

	if isinstance(f, pd.Series):
		f = f.values
	diff = np.nanmean(f[:5]) - np.nanmedian(f)

	return np.log(diff) + t0 / default_alpha


# Testing
# -------

def test_initialisation():
	import copy
	import george
	from . import kernels, detrender
	from ..__init__ import LOCAL_DATA
	from .. import lcf_tools

	k2_kernel = kernels.ClassicK2Kernel()
	kernel = k2_kernel.kernel

	ramp_model = RampModel(A_0=1)

	# Stage one (basic integration into george.GP)
	gp = george.GP(kernel,
				   mean=ramp_model,
				   white_noise=1.0,
				   fit_mean=True,
				   fit_white_noise=True)
	gp.get_parameter_names()
	gp.get_parameter_vector()
	print(gp.get_parameter_dict())
	gp.compute(np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]))
	gp.get_value(np.array([1, 1, 1]))
	gp.get_gradient(np.array([1, 1, 1]))
	gp.freeze_all_parameters()
	gp.thaw_all_parameters()
	print(ramp_model.unfrozen_mask)
	gp.freeze_parameter('mean:y_offset')
	print(ramp_model.unfrozen_mask)

	# Stage two (integration into K2Detrender)
	ramp_model = copy.deepcopy(ramp_model)
	del gp

	lcf = pd.read_pickle('{}/trappist/k2gp200164267-c12-lcf-full.pickle'.format(LOCAL_DATA))
	lcf['x'] = lcf.x_pos
	lcf['y'] = lcf.y_pos
	lcf = lcf_tools.initialise_lcf(lcf, f_col='f_tpf_track')

	k2_detrender = detrender.K2Detrender(lcf, k2_kernel, ramp=True)

	return gp



