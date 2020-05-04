"""Defines the PseudoVoigt1D model using an astropy.modeling.FittableModel."""

import math

import numpy as np
from astropy.modeling import Fittable1DModel, Parameter

__author__ = "Jeffrey Simpson"
__copyright__ = "Copyright 2020, Jeffrey Simpson"
__credits__ = ["Jeffrey Simpson"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jeffrey Simpson"
__email__ = "jeffrey.simpson@unsw.edu.au"
__status__ = "Development"


def tie_ca_lines_1(model):
    """Tie the second CaT line wavelength to the first."""
    mean = 8542.09 - 8498.03 + model.x_0_0
    return mean


def tie_ca_lines_2(model):
    """Tie the third CaT line wavelength to the first."""
    mean = 8662.14 - 8498.03 + model.x_0_0
    return mean


FLOAT_EPSILON = float(np.finfo(np.float32).tiny)


class PseudoVoigt1D(Fittable1DModel):
    """
    One dimensional Pseudo-Voigt model.

    Parameters
    ----------
    amplitude : float
        Amplitude of the Pseudo-Voigt.
    x_0 : float
        Mean of the Pseudo-Voigt.
    gamma_L : float
        Standard deviation of the Lorentzian.
    gamma_G : float
        Standard deviation of the Gaussian.

    Notes
    -----
    Using function has defined by Thompson et al (1987)
    DOI: 10.1107/S0021889887087090

    """

    x_0 = Parameter(default=0)
    # Ensure gamma_X makes sense if their bounds are not explicitly set.
    # gamma_X must be non-zero and positive.
    gamma_L = Parameter(default=1, bounds=(FLOAT_EPSILON, None))
    gamma_G = Parameter(default=1, bounds=(FLOAT_EPSILON, None))
    amplitude = Parameter(default=1, bounds=(FLOAT_EPSILON, None))

    @staticmethod
    def evaluate(x, x_0, gamma_L, gamma_G, amplitude):
        """Calculate the pseudo-Voigt function."""
        Γ_G = 2*math.sqrt(math.log1p(2))*gamma_G
        Γ_L = 2*gamma_L
        Γ_int = (Γ_G**5 +
                 2.69269 * Γ_G**4 * Γ_L**1 +
                 2.42843 * Γ_G**3 * Γ_L**2 +
                 4.47163 * Γ_G**2 * Γ_L**3 +
                 0.07842 * Γ_G**1 * Γ_L**4 +
                 Γ_L**5)
        Γ = np.power(Γ_int, 1/5)
        η = (1.36603 * (Γ_L/Γ) -
             0.47719 * (Γ_L/Γ)**2 +
             0.11116 * (Γ_L/Γ)**3)
        G_x = ((1/(math.sqrt(np.pi)*gamma_G)) *
               np.exp((-1*np.power(x-x_0, 2)) / (gamma_G**2)))
        L_x = gamma_L / (np.pi * (np.power(x-x_0, 2) + gamma_L**2))
        return amplitude*(η*L_x + (1-η)*G_x)


class ThreePseudoVoigts(PseudoVoigt1D + PseudoVoigt1D + PseudoVoigt1D):
    """Evaluates the sum of three PseudoVoigt1D."""
