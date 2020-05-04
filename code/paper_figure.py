#!/usr/bin/env python

"""paper_figure.py: Makes the figure for the research note."""



__author__ = "Jeffrey Simpson"
__copyright__ = "Copyright 2020, Jeffrey Simpson"
__credits__ = ["Jeffrey Simpson"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jeffrey Simpson"
__email__ = "jeffrey.simpson@unsw.edu.au"
__status__ = "Development"


import load_gaia_table
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


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


def actual_plotting(ax, xvals, yvals, yerrs, idx, label, kwargs):
    """Do the actual plotting."""
    ax.errorbar(xvals[idx], yvals[idx],
                yerr=yerrs[idx],
                elinewidth=0.5,
                lw=0.5,
                label=label, **kwargs)


cluster_params = load_gaia_table.cluster_params_table()

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

cluster_to_plot = everything_list.cluster_name.unique()

bp_rp_dered = everything_list.bp_rp_dered.values
V_clusters = everything_list.abs_G_mag.values
EW_clusters = everything_list.sum_ew_med.values
e_ew_clusters = everything_list.sum_ew_p.values
e_feh_clusters = everything_list.e_feh.values
FeH_clusters = everything_list.feh.values
median_table = everything_list.groupby('cluster_name').median()
everything_list['feh_calc'] = feh_calc(fitted_vals,
                                       V_clusters,
                                       EW_clusters,
                                       power)


sns.set_context("paper", font_scale=1.0)
fig, axes = plt.subplots(ncols=1, nrows=1,
                         figsize=(3.32*2, 3*2), constrained_layout=True,
                         sharey=True)
cm = matplotlib.cm.viridis
ax = axes

feh_range = [-2.6, -0.4]
color = feh_colour(everything_list.feh)
all_stars = ax.scatter(V_clusters, EW_clusters, s=0.5, c=everything_list.feh,
                       vmin=feh_range[1], vmax=feh_range[0],
                       alpha=1.0, cmap=matplotlib.cm.viridis)

for special_cluster in cluster_to_plot:
    cluster_idx = everything_list['cluster_name'] == special_cluster
    kwarg = dict(alpha=1.0, ms=3, mfc=color[cluster_idx][0], mec="None",
                 fmt='o', ecolor=color[cluster_idx][0])
    actual_plotting(ax, V_clusters, EW_clusters, e_ew_clusters,
                    cluster_idx, '__nolabel__', kwarg)

cbar = fig.colorbar(all_stars, orientation='horiztonal', ax=axes)

for feh_line in np.arange(feh_range[0], feh_range[1], 0.3):
    ax.plot(
        EW_G_curve(np.arange(1, 12, 0.1),
                   feh_line, fitted_vals, power),
        np.arange(1, 12, 0.1),
        lw=1, alpha=1.0, linestyle='--',
        color=all_stars.cmap(all_stars.colorbar.norm(feh_line)))


cbar.set_label("Literature [Fe/H]")
cbar.set_ticks(np.arange(-2.5, 0.5, 0.5))
axes.set_xlabel(r"$M_G$")
axes.set_ylabel(r"$\sum(\mathrm{EW}_\mathrm{CaT}) (\AA)$")

axes.set_xlim([2.6, -3.2])
axes.set_yticks(np.arange(2,11,2))
axes.set_ylim([1.5, 11])
plt.savefig("../paper/figures/ew_comparison.pdf",
            bbox_inches='tight')
plt.close('all')
# plt.show()
