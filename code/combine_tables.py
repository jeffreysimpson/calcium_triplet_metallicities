#!/usr/bin/env python

"""Combine all of the cluster tables into one supertable."""

import os

import numpy as np
import pandas as pd

__author__ = "Jeffrey Simpson"
__copyright__ = "Copyright 2019, Jeffrey Simpson"
__credits__ = ["Jeffrey Simpson"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jeffrey Simpson"
__email__ = "jeffrey.simpson@unsw.edu.au"
__status__ = "Development"

base_dir = "../data/spectra"

FIRST = True
for cluster_name in os.listdir(base_dir):
    if not os.path.isfile(
        f"{base_dir}/{cluster_name}/{cluster_name}_100_refined.out"):
        continue
    print(cluster_name)
    cluster_table = pd.read_csv(
        f"{base_dir}/{cluster_name}/{cluster_name}_100_refined.out")
    # This is to test that there is an AGB column
    print(np.sum(cluster_table.AGB))
    cluster_table['cluster_name'] = cluster_name
    if FIRST:
        big_table = cluster_table.copy()
        FIRST = False
    else:
        big_table = pd.concat([big_table, cluster_table])
big_table.to_csv(f"{base_dir}/all_cominbed.csv")
