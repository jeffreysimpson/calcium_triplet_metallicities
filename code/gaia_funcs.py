#!/usr/bin/env python

"""gaia_funcs.py: Contains a series of common functions for Gaia data."""

__author__ = "Jeffrey Simpson"
__copyright__ = "Copyright 2020, Jeffrey Simpson"
__credits__ = ["Jeffrey Simpson"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jeffrey Simpson"
__email__ = "jeffrey.simpson@unsw.edu.au"
__status__ = "Development"


def good_tmass_photom(table):
    """Requre stars to have A quality 2MASS J and Ks photometry."""
    return [(ph_qual[0] == 'A') and (ph_qual[2] == 'A')
            if type(ph_qual) is str else False for ph_qual in table.ph_qual]


def good_photom_idx(table, FITS=False):
    """Use the photometric quality criteria from Evans+2018."""
    if FITS:
        bp_rp_excess = table[1].data['phot_bp_rp_excess_factor']
        bp_rp = table[1].data['bp_rp']
    else:
        bp_rp_excess = table.phot_bp_rp_excess_factor
        bp_rp = table.bp_rp
    return ((bp_rp_excess <
             1.3 + 0.06*bp_rp**2) &
            (bp_rp_excess >
             1.0 + 0.015*bp_rp**2))


def good_astrom_idx(table, FITS=False):
    """Require the star to have good astrometry."""
    if FITS:
        ruwe = table[1].data['ruwe']
    else:
        ruwe = table.ruwe
    return ruwe < 1.4


def k_calc(BP_RP, A_0, c_array):
    """Calculate the extinction coefficient.

    Parameters
    ----------
    BP_RP   : array
              The Gaia BP-RP colour of the target(s)
    A_0     : float
              The extinction coefficient
    c_array : list
              Parameters used to derive the Gaia extinction
              coefficients as a function of colour and extinction

    Returns
    ----------
    The ratio of A_X/A_0 for a given BP-RP colour.

    References
    ----------
    Babusiaux et al (2018) 10.1051/0004-6361/201832843
    """
    return (c_array[0] +
            c_array[1]*BP_RP +
            c_array[2]*BP_RP**2 +
            c_array[3]*BP_RP**3 +
            c_array[4]*A_0 +
            c_array[5]*A_0**2 +
            c_array[6]*BP_RP*A_0)


def m_red_correction(BP_RP, ebv, m_M_V, UNREDDENED=False):
    """Calculate the distance modulus and reddening in Gaia photometry.

    Parameters
    ----------
    BP_RP      : array
                 The Gaia BP-RP colour of the target(s)
    ebv        : float
                 The E(B-V) to be converted to E(BP-RP)
    m_M_V      : float
                 The (m-M)_V to be converted to (m-M)_G
    UNREDDENED : bool
                 Is the provided distance modulus m_M_V or m_M_0?

    Returns
    ----------
    ebr   : array
            The E(BP-RP) values for all the input values of BP_RP
    m_M_G : array
            The (m-M)_G value for all the input values of BP_RP

    References
    ----------
    Babusiaux et al (2018) 10.1051/0004-6361/201832843
    """

    c_B = [1.1517, -0.0871, -0.0333, 0.0173, -0.0230, 0.0006, 0.0043]
    c_R = [0.6104, -0.0170, -0.0026, -0.0017, -0.0078, 0.00005, 0.0006]
    c_G = [0.9761, -0.1704, 0.0086, 0.0011, -0.0438, 0.0013, 0.0099]

    A_0 = 3.1*ebv
    # This checks if we are getting the (m-M)_0 or (m-M)_V
    if not UNREDDENED:
        m_M_0 = m_M_V - A_0
    else:
        m_M_0 = m_M_V
    k_B = k_calc(BP_RP, A_0, c_B)
    k_R = k_calc(BP_RP, A_0, c_R)
    k_G = k_calc(BP_RP, A_0, c_G)
    ebr = A_0 * (k_B - k_R)
    A_G = A_0 * k_G
    m_M_G = m_M_0 + A_G
    return ebr, m_M_G, A_G


def gaia_correction(table):
    """Correction to G magnitudes published in Gaia DR2."""
    # https://www.cosmos.esa.int/web/gaia/dr2-known-issues
    correction = np.ones(len(table)) * 0.032
    idx_pairs = [[(table.phot_g_mean_mag > 6) & (table.phot_g_mean_mag <= 16),
                  0.0032*(table.phot_g_mean_mag-6)]]
    for idx_pair in idx_pairs:
        correction[idx_pair[0]] = idx_pair[1][idx_pair[0]]
    return correction
