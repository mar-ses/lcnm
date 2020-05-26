"""GP model objects and tools, for ASTROphysical NOise Modelling.

CONVENTION:
    - pass around direct np.arrays, not lcf DataFrames;
        i.e X = lcf[['t', 'x', 'y']].values
            f = lcf.f.values
        The object here don't know how lcf objects work, but the
        convention will be that the dependent variable is f, while
        the independent variables are:  [x, y,] t

STRUCTURE: 
TODO: see if the following needs to be reassessed.
    - Aim for a structure where we separate Kernel and GPDetrender objects.
    - GPDetrender objects may even be general for a kernel
    - Kernel must contain the GP kernels, as well as DE bounds and priors.
    - GPDetrender object must be able to calculate the lnposterior
    - GPDetrender object must also take care of the full detrending
    - GPDetrender contains the data only through GP

POSSIBILITIES:
    - 

EXCEPTIONS:
    NonPositiveDefiniteError
    OversamplingPopulationError
    PriorInitialisationError
    LAPACKError

NOTE on short cadence extension (call it BTS - or big time-series):
    - Will create an SCGPDetrender (or SCDetrended) which will overload
        the necessary methods with the optimized short-cadence
        algorithms.

NOTE :
    - TODO: go back and look at structure again. It is getting a little
        bloated and "general." Keep this in terms of very specific objects
        that to very specific jobs. I.e one is a prior/kernel (switch to
        prior perhaps). Another is a GP wrapper, with detrend methods,
        and lnlikelihood/lnprior exposure; possibly two different
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

from .kernels import (ClassicK2Kernel, QuasiPeriodicK2Kernel,
                      QuasiPeriodic1DKernel, Matern52K2Kernel)
from .gp_model_base import (LCNoiseModel1D,
                            NonPositiveDefiniteError,
                            OversamplingPopulationError,
                            PriorInitialisationError,
                            LAPACKError)
from .gp_model_3d import LCNoiseModel3D
from .gp_model_long import LLCNoiseModel1D, LLCNoiseModel3D
from .model_objects import RampModel

