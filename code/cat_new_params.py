#!/usr/bin/env python

"""cat_new_params.py: Find relationship in Gaia photometry for Fe from EW.

This code in part uses code by DFM from
https://emcee.readthedocs.io/en/latest/tutorials/line/ (MIT License)

The code will either use hardcoded values or will compute them using emcee.
Change the mcmc value to True to do this.
"""

import argparse
import sys
from os.path import basename

import corner
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import emcee
import gaia_funcs
import load_gaia_table

__author__ = "Jeffrey Simpson"
__copyright__ = "Copyright 2020, Jeffrey Simpson"
__credits__ = ["Jeffrey Simpson"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jeffrey Simpson"
__email__ = "jeffrey.simpson@unsw.edu.au"
__status__ = "Development"


def feh_colour(FeH):
    """Determine colour for a given metallicity."""
    return cm(1/(feh_range[1]-feh_range[0]) *
              (FeH - feh_range[0]))


def feh_calc(theta, V, EW, power):
    """Calculate the metallicity for a given magnitude and EW."""
    a, b, c, d, e = theta
    # Wdash = reduced_ew(theta, V, EW)
    FeH = a + b*V + c*EW + d*EW**power + e*V*EW
    # FeH = d + f*np.power(Wdash, 1) + g*np.power(Wdash, 2)
    return FeH


def EW_G_curve(EW, FeH, fitted_vals, power):
    """Calculate curve in V/EW space for a given metallicity."""
    a, b, c, d, e = fitted_vals
    return -1*(a - FeH + d*EW**(power) + c*EW)/(b + e*EW)


def log_likelihood(theta, V, EW, FeH, e_ew, e_feh, power):
    """Calculate the log likelihood function."""
    model = feh_calc(theta, V, EW, power)
    sigma2 = e_ew**2 + e_feh**2
    return -0.5*np.sum((FeH-model)**2/sigma2) + np.log(sigma2)


def log_prior(theta):
    """Not assuming any priors."""
    return 0.0


def log_probability(theta, V, EW, FeH, e_ew, e_feh, power):
    """Calculate the log probability."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, V, EW, FeH, e_ew, e_feh, power)


def do_mcmc(fitted_vals):
    ndim = len(fitted_vals)

    # We'll sample with 250 walkers.
    nwalkers = 250

    # Choose an initial set of positions for the walkers.
    p0 = [fitted_vals + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(V_clusters[~eso280_idx],
                                          EW_clusters[~eso280_idx],
                                          FeH_clusters[~eso280_idx],
                                          e_ew_clusters[~eso280_idx],
                                          e_feh_clusters[~eso280_idx],
                                          power))
    sampler.run_mcmc(p0, 1000, progress=True)

    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["a", "b", "c", "d", "e"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.05)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.1)
    axes[-1].set_xlabel("step number")
    fig.savefig(
        "../paper/figures/mcmc_chain.pdf", bbox_inches="tight")
    plt.close('all')

    flat_samples = sampler.get_chain(discard=400, thin=25, flat=True)
    print(flat_samples.shape)

    fitted_vals_new = []
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "{0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1])
        print(txt)
        fitted_vals_new.append(mcmc[1])

    fig = corner.corner(flat_samples, labels=labels, truths=fitted_vals)
    fig.savefig(
        "../paper/figures/mcmc_corner.pdf", bbox_inches="tight")
    plt.close('all')

    fitted_vals = fitted_vals_new
    print(fitted_vals)
    return fitted_vals


def right_plot_test(xy):
    return (plot_dict['x'] == xy[0]) & (plot_dict['y'] == xy[1])


parser = argparse.ArgumentParser(
    description="Processes the CaT spectral data.",
    usage=f"{basename(__file__)} -m <True/False>")
parser.add_argument('--run_mcmc', dest='run_mcmc', action='store_true')
parser.add_argument('--no_mcmc', dest='run_mcmc', action='store_false')
parser.set_defaults(run_mcmc=False)
parser.add_argument('-c', '--cluster_to_plot',
                    default='all',
                    help="Cluster name to process")

# If someone types the name of the program with no arguments, let's
# have argparse print the help message and quit.
#
# The full command + argument list can be read from sys.argv.
# The arguments are sys.argv[1:] (the name of the program is sys.argv[0].
#
if len(sys.argv[1:]) == 0:
    print()
    parser.print_help()
    parser.exit()
    print()

args = parser.parse_args()
run_mcmc = args.run_mcmc
cluster_to_plot = args.cluster_to_plot

# These values have been pre-determined for the data.
fitted_vals = [-3.5238827799651524, 0.10836044995877069, 0.4102658220313341,
               -0.006607142064140818, 0.014668538469726208]
power = 2
a, b, c, d, e = fitted_vals

# Load data from other clusters
base_dir = "../data/spectra"
everything_list = load_gaia_table.cluster_table(f"{base_dir}/all_cominbed.csv",
                                                LOAD_AGB=False)
cluster_params = load_gaia_table.cluster_params_table()
# We only care about stars that are brighter than about the HB.
everything_list = everything_list[everything_list.abs_G_mag < 2.5]
eso280_idx = everything_list.cluster_name == "ESO280"
if cluster_to_plot == 'everything':
    cluster_to_plot = everything_list.cluster_name.unique()
else:
    cluster_to_plot = [cluster_to_plot]

bp_rp_dered = everything_list.bp_rp_dered.values
V_clusters = everything_list.abs_G_mag.values
EW_clusters = everything_list.sum_ew_med.values
e_ew_clusters = everything_list.sum_ew_p.values
e_feh_clusters = everything_list.e_feh.values
FeH_clusters = everything_list.feh.values
if run_mcmc:
    fitted_vals = do_mcmc(fitted_vals)
median_table = everything_list.groupby('cluster_name').median()
everything_list['feh_calc'] = feh_calc(fitted_vals,
                                       V_clusters,
                                       EW_clusters,
                                       power)
# reduced_ew_values = everything_list.reduced_ew.values
feh_calc_values = everything_list.feh_calc.values
median_table = everything_list.groupby('cluster_name').median()


xy_dicts = {"V_clusters": {"values": V_clusters,
                           "label": r'$M_G$',
                           "lims": [3.5, -3.5],
                           "ticks": np.arange(-3, 4.0, 1)},
            "EW_clusters": {"values": EW_clusters,
                            "label": r"EW $(\AA)$",
                            "lims": [1, 11],
                            "ticks": None},
            "FeH_clusters": {"values": FeH_clusters,
                             "label": r"[Fe/H]$_\mathrm{Lit}$",
                             "lims": [-2.7, -0.25],
                             "ticks": np.arange(-2.5, -0.4, 0.5)},
            "FeH_clusters_jitted": {"values": (FeH_clusters +
                                               np.random.normal(
                                                   scale=0.005,
                                                   size=len(FeH_clusters))),
                                    "label": r"[Fe/H]$_\mathrm{Lit}$",
                                    "lims": [-2.7, -0.25],
                                    "ticks": np.arange(-2.5, -0.4, 0.5)},
            "FeH_calc": {"values": feh_calc_values,
                         "label": r"[Fe/H]$_\mathrm{CaT}$",
                         "lims": [-2.7, -0.25],
                         "ticks": np.arange(-2.5, -0.4, 0.3)},
            "feh_diff": {"values": (FeH_clusters - feh_calc_values),
                         "label": r"[Fe/H]$_\mathrm{Lit}$ - [Fe/H]$_\mathrm{CaT}$",
                         "lims": [-0.6, +0.6],
                         "ticks": None},
            "bp_rp": {"values": bp_rp_dered,
                      "label": r"$G_\mathrm{BP} - G_\mathrm{RP}$",
                      "lims": [0.5, 2.5],
                      "ticks": None},
            "Gmag": {"values": V_clusters,
                     "label": r"$G$",
                     "lims": [4, -3.5],
                     "ticks": None},
            # "med_reduced_ew": {"values": median_table.reduced_ew},
            "med_feh_lit": {"values": median_table.feh},
            "med_feh_cat": {"values": median_table.feh_calc},
            "med_diff_feh": {"values": median_table.feh-median_table.feh_calc}}

plotting_dicts = [{"x": "V_clusters", "y": "EW_clusters"},
                  {"x": "V_clusters", "y": "FeH_calc"},
                  {"x": "bp_rp", "y": "Gmag"},
                  {"x": "FeH_calc", "y": "FeH_clusters_jitted",
                   "x_med": "med_feh_cat", "y_med": "med_feh_lit"},
                  {"x": "FeH_clusters_jitted", "y": "feh_diff",
                   "x_med": "med_feh_lit", "y_med": "med_diff_feh"},
                  {"x": "V_clusters", "y": "feh_diff"}]

sns.set_context("paper", font_scale=0.7)
cm = matplotlib.cm.get_cmap('jet')


feh_range = [min(everything_list.feh.unique()),
             max(everything_list.feh.unique())]
color = feh_colour(everything_list.feh)
G_array = np.arange(-3.0, 4, 0.1)
ew_array = np.arange(2, 10, 0.1)
for cluster_name in cluster_to_plot:
    if cluster_name != "all":
        cluster_idx = everything_list.cluster_name == cluster_name
        gaia_photom = pd.read_csv(
            f"../data/spectra/{cluster_name}/{cluster_name}_gaia_photometry.csv")
        gaia_photom = gaia_photom[gaia_photom.member]
        ebr, m_M_G = gaia_funcs.m_red_correction(
            gaia_photom.bp_rp,
            cluster_params[cluster_name][1],
            cluster_params[cluster_name][0])
    else:
        cluster_idx = np.ones(len(everything_list), dtype=np.bool)
        gaia_photom = []
    if np.sum(cluster_idx) == 0:
        print("No useful stars")
        continue
    fig, axes = plt.subplots(nrows=2, ncols=3,
                             figsize=(3.32*3, 5.2*1.3),
                             constrained_layout=True)
    for ax_count, plot_dict in enumerate(plotting_dicts):
        alpha = 0.7
        ax = axes.flatten()[ax_count]
        if cluster_name != "all":
            ax.scatter(xy_dicts[plot_dict['x']]["values"][~cluster_idx],
                       xy_dicts[plot_dict['y']]["values"][~cluster_idx],
                       c=color[~cluster_idx], marker='o',
                       s=1, alpha=0.1, lw=0)
            main_colour = 'k'
        else:
            main_colour = color[cluster_idx]
        ax.scatter(xy_dicts[plot_dict['x']]["values"][eso280_idx],
                   xy_dicts[plot_dict['y']]["values"][eso280_idx],
                   c='C3', marker='s', s=10, alpha=0.7, lw=0)
        ax.scatter(xy_dicts[plot_dict['x']]["values"][cluster_idx],
                   xy_dicts[plot_dict['y']]["values"][cluster_idx],
                   c=main_colour, marker='o', s=4, alpha=alpha, lw=0)
        if "x_med" in plot_dict:
            ax.scatter(xy_dicts[plot_dict['x_med']]["values"],
                       xy_dicts[plot_dict['y_med']]["values"],
                       marker='*', s=40, alpha=0.5, lw=0.2, c='k')

        # Extra bits on the plots
        for feh_line in np.arange(-2.6, 0., 0.3):
            if right_plot_test(["V_clusters", "EW_clusters"]):
                ax.plot(
                    EW_G_curve(np.arange(1, 12, 0.1),
                               feh_line, fitted_vals, power),
                    np.arange(1, 12, 0.1),
                    lw=1, alpha=0.3, color=feh_colour(feh_line))
            if right_plot_test(["V_clusters", "FeH_calc"]):
                ax.axhline(feh_line,
                           lw=1, alpha=0.3, color=feh_colour(feh_line))
        if right_plot_test(["V_clusters", "EW_clusters"]) & (cluster_name != "all"):
            ax.plot(EW_G_curve(np.arange(1, 12, 0.1),
                               everything_list[cluster_idx].feh.unique(),
                               fitted_vals, power),
                    np.arange(1, 12, 0.1), lw=1, alpha=0.6, color='k')
        if right_plot_test(["V_clusters", "FeH_calc"]) & (cluster_name != "all"):
            ax.axhline(everything_list[cluster_idx].feh.unique(),
                       lw=1, alpha=0.6, color='k')
        if (right_plot_test(["bp_rp", "Gmag"]) & (cluster_name != "all")):
            ax.scatter(
                gaia_photom[gaia_funcs.good_photom_idx(gaia_photom)].bp_rp-ebr,
                (gaia_photom[gaia_funcs.good_photom_idx(gaia_photom)].phot_g_mean_mag - m_M_G),
                marker='.', s=5, alpha=0.5, zorder=0, lw=0)
        if right_plot_test(["FeH_calc", "FeH_clusters_jitted"]):
            ax.plot([-2.6, 0.], [-2.6, 0.], lw=0.5, alpha=0.5, c='k')

        ax.set_xlim(xy_dicts[plot_dict['x']]['lims'])
        ax.set_ylim(xy_dicts[plot_dict['y']]['lims'])
        ax.set_xlabel(xy_dicts[plot_dict['x']]['label'])
        ax.set_ylabel(xy_dicts[plot_dict['y']]['label'])
    fig.savefig(
        f"../paper/figures/cat_ew_GCs_{cluster_name}_pv_{power}.pdf",
        bbox_inches="tight")
    plt.close('all')
#     plt.show()
