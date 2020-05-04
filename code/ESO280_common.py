#!/usr/bin/env python

"""ESO280_common.py: Central place for common values and selections."""

import astropy.units as u
import numpy as np
import pandas as pd
from astropy import uncertainty as unc
from astropy.coordinates import SkyCoord
from astropy import constants as const
import gaia_funcs

__author__ = "Jeffrey Simpson"
__copyright__ = "Copyright 2019, Jeffrey Simpson"
__credits__ = ["Jeffrey Simpson"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jeffrey Simpson"
__email__ = "jeffrey.simpson@unsw.edu.au"
__status__ = "Development"

cluster_centre = SkyCoord(ra=272.27612983712135*u.degree,
                          dec=-46.422586369434775*u.degree)


def load_table(DROP_DUPLICATES=True, PHOTOM_CORRECTION=True, TMASS=False):
    """Load the data table. By default this removes twice observed stars."""
    if TMASS:
        # Use the table with 2MASS photometry.
        observed = pd.read_csv("../data/ESO280_100_unrefined_hb_tmass.out")
    else:
        observed = pd.read_csv("../data/ESO280_100_unrefined_hb.out")
    if PHOTOM_CORRECTION:
        observed['phot_g_mean_mag'] -= gaia_funcs.gaia_correction(observed)
    if DROP_DUPLICATES:
        return observed.sort_values('snr', ascending=False).drop_duplicates(
            'source_id', keep='first')
    else:
        return observed


def ESO280_params(PRINT=True):
    params_dict = {"eso_280_m_M_V": 17.011,
                   "eso_280_e_m_M_V": 0.045,
                   "eso_280_ebv": 0.141,
                   "eso_280_e_ebv": 0.006,
                   "rv": 94.644,
                   "e_rv": 0.476,
                   "std_rv": 2.305,
                   "e_std_rv": 0.363,
                   "r_t": [0.14693*u.deg, 0.04126*u.deg],
                   "r_c": [0.00410*u.deg, 0.00009*u.deg]}
    n_samples = 10000
    eso_280_m_M_V_dist = unc.normal(params_dict['eso_280_m_M_V'],
                                    std=params_dict['eso_280_e_m_M_V'],
                                    n_samples=n_samples)
    eso_280_ebv_dist = unc.normal(params_dict['eso_280_ebv'],
                                  std=params_dict['eso_280_e_ebv'],
                                  n_samples=n_samples)
    eso_280_m_M_0_dist = eso_280_m_M_V_dist - 3.1*eso_280_ebv_dist
    eso_280_dist_dist = unc.Distribution(
        10**(1+eso_280_m_M_0_dist/5).distribution*u.pc)

    # Hardcoded values. Calculated using velocity_estimate.py
    rv_dist = unc.normal(params_dict['rv']*u.km/u.s,
                         std=params_dict['e_rv']*u.km/u.s,
                         n_samples=n_samples)
    rv_std_dist = unc.normal(params_dict['std_rv']*u.km/u.s,
                             std=params_dict['e_std_rv']*u.km/u.s,
                             n_samples=10000)
    # Size values from ASteCA
    r_0_dist = unc.normal(params_dict['r_c'][0],
                          std=params_dict['r_c'][1],
                          n_samples=10000)
    r_t_dist = unc.normal(params_dict['r_t'][0],
                          std=params_dict['r_t'][1],
                          n_samples=10000)
    size_dist = (np.tan(r_0_dist) * eso_280_dist_dist)
    tidal_dist = (np.tan(r_t_dist) * eso_280_dist_dist)
    cluster_mass = ((7.5 * rv_std_dist**2 * 4/3*size_dist) / const.G)

    sc_best = SkyCoord(
        ra=cluster_centre.ra,
        dec=cluster_centre.dec,
        radial_velocity=rv_dist.pdf_mean(),
        distance=eso_280_dist_dist.pdf_mean(),
        pm_ra_cosdec=-0.548*u.mas/u.yr,
        pm_dec=-2.688*u.mas/u.yr
        )
    eso_280_pmra_dist = unc.normal(sc_best.pm_ra_cosdec,
                                   std=0.073*u.mas/u.yr,
                                   n_samples=n_samples)
    eso_280_pmdec_dist = unc.normal(sc_best.pm_dec,
                                    std=0.052*u.mas/u.yr,
                                    n_samples=n_samples)
    sc_dist = SkyCoord(
        ra=np.ones(eso_280_dist_dist.n_samples)*cluster_centre.ra,
        dec=np.ones(eso_280_dist_dist.n_samples)*cluster_centre.dec,
        radial_velocity=rv_dist.distribution,
        distance=eso_280_dist_dist.distribution,
        pm_ra_cosdec=eso_280_pmra_dist.distribution,
        pm_dec=eso_280_pmdec_dist.distribution)
    if PRINT:
        print(
            rf"$r_c$ & ${params_dict['r_c'][0].to(u.arcsec).value:0.2f}\pm{params_dict['r_c'][1].to(u.arcsec).value:0.2f}$~arcsec\\")
        print(
            rf"$r_t$ & ${params_dict['r_t'][0].to(u.arcmin).value:0.2f}\pm{params_dict['r_t'][1].to(u.arcmin).value:0.2f}$~arcmin\\")
        print(
            rf"$(m-M)_V$ & ${params_dict['eso_280_m_M_V']:0.2f}\pm{params_dict['eso_280_e_m_M_V']:0.2f}$\\")
        print(
            rf"$\ebv$ & ${params_dict['eso_280_ebv']:0.2f}\pm{params_dict['eso_280_e_ebv']:0.2f}$\\")
        print(
            rf"$(m-M)_0$ & ${eso_280_m_M_0_dist.pdf_mean:0.2f}\pm{eso_280_m_M_0_dist.pdf_std:0.2f}$\\")
        print(
            rf"$d_\odot$ & ${eso_280_dist_dist.pdf_mean.to(u.kpc).value:0.1f}\pm{eso_280_dist_dist.pdf_std.to(u.kpc).value:0.1f}$~kpc\\")
        print(
            rf"$r_c$ & ${size_dist.pdf_mean.to(u.pc).value:0.2f}\pm{size_dist.pdf_std.to(u.pc).value:0.2f}$~pc\\")
        print(
            rf"$r_t$ & ${tidal_dist.pdf_mean.to(u.pc).value:0.1f}\pm{tidal_dist.pdf_std.to(u.pc).value:0.1f}$~pc\\")
        print(rf"Mass & $({cluster_mass.pdf_mean.to(u.solMass).value/1000:0.1f}\pm{cluster_mass.pdf_std.to(u.solMass).value/1000:0.1f})\times10^3$~M$_\odot$\\")
        print(rf"$v_r$ & ${params_dict['rv']:0.2f}\pm{params_dict['e_rv']:0.2f}$\kms\\")
        print(
            rf"$\sigma_r$ & ${params_dict['std_rv']:0.2f}\pm{params_dict['e_std_rv']:0.2f}$\kms\\")
    return params_dict, sc_best, sc_dist


def ESO280_idxs(observed):
    """Select for various groups of stars in ESO280."""
    params_dict, *_ = ESO280_params(PRINT=False)
    ang_distance = params_dict["r_t"][0]
    c_observed = SkyCoord(ra=np.array(observed.ra)*u.degree,
                          dec=np.array(observed.dec)*u.degree)

    close_observed_idx = c_observed.separation(cluster_centre) < ang_distance
    # low_ew_idx = observed.sum_ew_med < 5.
    rv_idx = ((observed.true_rv_med > 95-25.) &
              (observed.true_rv_med < 95+25.))
    pm_idx = np.sqrt((observed.pmra--0.548)**2 +
                     (observed.pmdec--2.688)**2) < 1.5
    ew_G_idx = -0.5*observed.phot_g_mean_mag + 12.2 > observed.sum_ew_med
    # color_select_idx = observed.bp_rp > 0.8
    hb_idx = observed.num_good == 200
    rgb_possible_idx = ew_G_idx & rv_idx & ~hb_idx & pm_idx
    hb_possible_idx = rv_idx & hb_idx & pm_idx

    members_not_hb_idx = rgb_possible_idx & close_observed_idx
    et_not_hb_idx = rgb_possible_idx & ~close_observed_idx

    members_hb_idx = hb_possible_idx & close_observed_idx
    et_hb_idx = hb_possible_idx & ~close_observed_idx

    # extra_tidal_idx = np.in1d(observed.source_id,
    #                           [6719718983068881664,
    #                            6719556186624323456,
    #                            # 6719616316170550528, # 6719903155571676544,
    #                            6719533204255711488,
    #                            6719866626860277760])
    cn_star_idx = np.in1d(observed.source_id, [6719598900092253184])
    ch_star_idx = np.in1d(observed.source_id, [6719599101938996864])

    idx_dict = {"close_observed_idx": close_observed_idx,
                # "low_ew_idx": low_ew_idx,
                "rv_idx": rv_idx,
                "members_not_hb_idx": members_not_hb_idx,
                "members_hb_idx": members_hb_idx,
                "et_not_hb_idx": et_not_hb_idx,
                "et_hb_idx": et_hb_idx,
                # "extra_tidal_idx": extra_tidal_idx,
                "cn_star_idx": cn_star_idx,
                "ch_star_idx": ch_star_idx,
                "hb_idx": hb_idx}

    plot_dict = [{"idx": (~(members_not_hb_idx | members_hb_idx |
                            hb_idx | et_not_hb_idx | et_hb_idx) &
                          ~close_observed_idx),
                  "label": r"$>8.8$ arcmin field",
                  "kwargs": dict(alpha=0.3, ms=3, mfc=[0.6, 0.6, 0.6],
                                 mec="None", fmt='.')},
                 {"idx": (~(members_not_hb_idx | members_hb_idx | hb_idx) &
                          close_observed_idx &
                          ((observed.num_good == 200) |
                           (observed.sum_ew_p < 2.0))),
                  "label": r"$<8.8$ arcmin field",
                  "kwargs": dict(alpha=0.5, ms=4, mfc='k',
                                 mec="None", fmt='.')},
                 {"idx": hb_idx & ~members_hb_idx,
                  "label": "__nolabel__",
                  "kwargs": dict(alpha=0.3, ms=3, mfc=[0.6, 0.6, 0.6],
                                 mec="None", fmt='.')},
                 {"idx": members_not_hb_idx & ~(cn_star_idx),
                  "label": "Members",
                  "kwargs": dict(alpha=0.8, ms=5, mfc='C3',
                                 mec="None", fmt='o')},
                 {"idx": members_hb_idx & ~(cn_star_idx),
                  "label": "__nolabel__",
                  "kwargs": dict(alpha=0.8, ms=5, mfc='C3',
                                 mec="None", fmt='o')},
                 {"idx": cn_star_idx,
                  "label": "CN-strong star",
                  "kwargs": dict(alpha=0.8, ms=15, mfc='C0',
                                 mec="None", fmt='*', zorder=1000)},
                 {"idx": et_not_hb_idx | et_hb_idx,
                  "label": "Extra-tidal stars",
                  "kwargs": dict(alpha=0.8, ms=7, mfc='C2',
                                 mec="None", fmt='s')}]
    return idx_dict, plot_dict
