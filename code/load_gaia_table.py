import pandas as pd
import numpy as np
import gaia_funcs
import ESO280_common


def cluster_table(file_name, LOAD_AGB=False, DROP_DUPLICATES=True):
    """Load cluster table and add FeH, distance and reddening info."""
    full_cluster_table = pd.read_csv(file_name)
    print(len(full_cluster_table))
    if not LOAD_AGB:
        full_cluster_table = full_cluster_table[~full_cluster_table['AGB']]
    print(len(full_cluster_table))
    ignore_table = pd.read_csv("../../CaT_Metallicity/data/ignore_stars.csv")
    bad_sources = full_cluster_table[np.in1d(full_cluster_table.source_id,
                                             ignore_table.source_id)]
    useful_clusters = full_cluster_table[~np.in1d(full_cluster_table.index,
                                                  bad_sources[np.in1d(
                                                      bad_sources.file_name,
                                                      ignore_table.file_name
                                                      )].index)].copy()
    useful_clusters.phot_g_mean_mag -= gaia_funcs.gaia_correction(useful_clusters)
    # useful_clusters = full_cluster_table.copy()
    useful_clusters['abs_G_mag'] = np.nan
    useful_clusters['feh'] = np.nan
    useful_clusters['e_feh'] = 0.05
    useful_clusters['ew'] = useful_clusters['sum_ew_med']
    useful_clusters['e_ew'] = useful_clusters[['sum_ew_p',
                                               'sum_ew_n']].max(axis=1)
    cluster_params = cluster_params_table()
    for cluster, cluster_subtable in useful_clusters.groupby('cluster_name'):
        try:
            ebr, m_M_G, *_ = gaia_funcs.m_red_correction(
                cluster_subtable.bp_rp,
                cluster_params[cluster][1],
                cluster_params[cluster][0])
            useful_clusters.loc[
                cluster_subtable.index,
                'abs_G_mag'] = cluster_subtable.phot_g_mean_mag - m_M_G
            useful_clusters.loc[
                cluster_subtable.index,
                'bp_rp_dered'] = cluster_subtable.bp_rp - ebr
            useful_clusters.loc[
                cluster_subtable.index,
                'feh'] = cluster_params[cluster][2]
        except KeyError:
            print(f"{cluster} not in dictionary")
            continue
    useful_clusters = useful_clusters[
        gaia_funcs.good_photom_idx(useful_clusters) &
        (~np.isnan(useful_clusters['abs_G_mag'])) &
        (useful_clusters['num_good'] > 90) &
        (useful_clusters['sum_ew_med'] < 10) &
        (useful_clusters['snr'] > 20)]
    if DROP_DUPLICATES:
        useful_clusters = useful_clusters.sort_values(
            'snr', ascending=False).drop_duplicates('source_id', keep='first')
    useful_clusters.reset_index(inplace=True, drop=True)
    useful_clusters = useful_clusters[~np.isnan(useful_clusters.feh)]
    return useful_clusters


def cluster_params_table():
    params_dict, *_ = ESO280_common.ESO280_params(PRINT=False)
    eso_280_m_M_V = params_dict['eso_280_m_M_V']
    eso_280_ebv = params_dict['eso_280_ebv']
    cluster_params = dict({
        'ESO280': [eso_280_m_M_V, eso_280_ebv, -2.5, True],
        'ESO452': [16.29, 0.61, -0.81, True],
        'IC4499': [17.07, 0.18, -1.53, True],  # Usher
        'NGC104': [13.37, 0.04, -0.72, True],  # Usher
        'NGC1851': [15.47, 0.02, -1.21, True],  # Usher
        'NGC1904': [15.59, 0.01, -1.55, True],  # Usher
        'NGC2298': [15.60, 0.14, -1.92, True],
        'NGC288': [14.84, 0.03, -1.26, True],  # Usher
        # 'NGC3201': [14.20, 0.24, -1.59, True],
        'NGC362': [14.83, 0.05, -1.15, True],  # Usher
        'NGC4590': [15.21, 0.05, -2.23, True],  # Usher
        'NGC5024': [16.32, 0.02, -2.04, True],  # Usher
        'NGC5053': [16.23, 0.01, -2.27, True],
        # 'NGC5927': [15.82, 0.45, -0.43, True],  # Usher
        # 'NGC5946': [16.79, 0.54, -1.29, True],
        # 'NGC6121': [12.82, 0.35, -1.01, True], #-1.16
        'NGC6144': [15.86, 0.36, -1.76, True],
        'NGC6218': [14.01, 0.19, -1.38, True],  # Usher
        'NGC6553': [15.83, 0.63, -0.16, True],  # Usher
        # 'NGC6558': [15.70, 0.44, -1.01, True],
        'NGC6624': [15.36, 0.28, -0.69, True],  # Usher
        # 'NGC6626': [14.95, 0.40, -1.29, True],  # Usher
        'NGC6637': [15.28, 0.18, -0.75, True],  # Usher
        'NGC6681': [14.99, 0.07, -1.63, True],  # Usher
        'NGC6752': [13.13, 0.04, -1.48, True],  # Usher
        'NGC6809': [13.89, 0.08, -1.89, True],  # Usher
#         'NGC7089': [15.50, 0.06, -1.53, True],  # Usher
        'NGC7099': [14.64, 0.03, -2.44, True],  # Usher
        'Pal5': [16.92, 0.03, -1.41, True],
        # 'Pal6': [18.34, 1.46, -0.91, True],
        # 'Terzan7': [17.01, 0.07, -0.32, True],
        'Terzan8': [17.47, 0.12, -2.06, True],  # -2.16
        'M67': [9.726, 0.03, -0.03, False],  # 9.65 0.04,
        'Melotte66': [13.63, 0.14, -0.38, True],
    })
    return cluster_params
