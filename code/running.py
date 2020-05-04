#!/usr/bin/env python

"""Process the spectra."""

import argparse
import logging
import os
import pickle
import re
import sys
import warnings
from os.path import basename

import astropy.units as u
import matplotlib
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from astropy.modeling import fitting
from astropy.utils.exceptions import AstropyWarning

from PseudoVoigt import ThreePseudoVoigts, tie_ca_lines_1, tie_ca_lines_2
from spectrum import Spectrum

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')


__author__ = "Jeffrey Simpson"
__copyright__ = "Copyright 2020, Jeffrey Simpson"
__credits__ = ["Jeffrey Simpson"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jeffrey Simpson"
__email__ = "jeffrey.simpson@unsw.edu.au"
__status__ = "Development"


def pickle_location(file_location):
    """Where is the pickle file meant to be."""
    save_dir = os.path.dirname(file_location) + "/pickles/"
    return f"{save_dir}{star.NAME}.pickle"


def outfile_initialize(cluster_name, base_dir, phot_table, suffix=""):
    """Open the outfile and write header info."""
    outfile = open(f"{base_dir}/{cluster_name}_{suffix}.out", 'w')
    for item in list(phot_table.columns):
        outfile.write("{},".format(item))

    for col_name in ['GRATANGL', 'CAMANGL', 'LAMBDAC',
                     'LAMBDAB', 'DISPERS', "arc_lamp",
                     "snr", "bary_corr_star", "UTMJD"]:
        outfile.write("{},".format(col_name))
    for col_name in ["ew_1", "ew_2", "ew_3", "sum_ew", "true_rv"]:
        for suffix in ["med", "p", "n"]:
            outfile.write("{}_{},".format(col_name, suffix))
        if col_name == "true_rv":
            outfile.write("num_good")
    outfile.write("\n")
    return outfile


def useful_star_info(outfile, star_spectrum):
    """Write info to output table."""
    outfile.write(f"{star_spectrum.star_name},")
    outfile.write(f"{star_spectrum.filename},")
    outfile.write(f"{star_spectrum.source_id},")
    outfile.write(f"{star_spectrum.star_position.ra.deg},")
    outfile.write(f"{star_spectrum.star_position.dec.deg},")
    for col_name in ['GRATANGL', 'CAMANGL', 'LAMBDAC',
                     'LAMBDAB', 'DISPERS']:
        outfile.write(f"{star_spectrum.header[col_name]},")
    outfile.write(f"{star_spectrum.arc_lamp},")
    outfile.write(f"{star_spectrum.snr},")
    outfile.write(f"{star_spectrum.barycorr.to(u.km/u.s).value},")
    outfile.write(f"{star_spectrum.header['UTMJD']}")


def percentile_range(star_data, outfile, rv=False, last=False):
    """Write out values of 16, 50, 86 percentiles."""
    if np.sum(~np.isnan(star_data)) == 0:
        outfile.write(",,,")
        return 0
    if rv:
        star_data = star_data.to(u.km/u.s).value
    p_value = np.nanpercentile(star_data, 84) - np.nanmedian(star_data)
    n_value = np.nanmedian(star_data) - np.nanpercentile(star_data, 16)
    outfile.write(",{}".format(np.nanmedian(star_data)))
    outfile.write(",{}".format(p_value))
    outfile.write(",{}".format(n_value))
    if last:
        outfile.write(",{}".format(np.sum(~np.isnan(star_data))))


parser = argparse.ArgumentParser(
    description="Processes the CaT spectral data.",
    usage=f"{basename(__file__)} -c <cluster name> -n <number_of_tests>")

parser.add_argument('-c', '--cluster_name',
                    required=True,
                    help="Cluster name to process")
parser.add_argument('-n', '--number_of_tests',
                    required=True,
                    type=int,
                    help="Number of tests to run")
parser.add_argument('-s', '--specific_star',
                    help="A specific star name to test",
                    default=None)
parser.add_argument('-d', '--base_dir',
                    help="Where the spectra is located.",
                    default="../data/spectra")
parser.add_argument('-S', '--split_point',
                    help="Start index for the splitting. -1 for no splitting.",
                    type=int,
                    default=-1)
parser.add_argument('--split_size',
                    help="How big are the chunks?",
                    type=int,
                    default=100)

parser.add_argument('--log_to_screen', dest='log', action='store_true')
parser.add_argument('--log_to_file', dest='log', action='store_false')
parser.set_defaults(log=False)

parser.add_argument('--plotting', dest='plotting', action='store_true')
parser.add_argument('--no_plotting', dest='plotting', action='store_false')
parser.set_defaults(log=False)

parser.add_argument('--overwrite', dest='overwrite', action='store_true')
parser.add_argument('--keep_pickles', dest='overwrite', action='store_false')
parser.set_defaults(log=False)

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
cluster_name = args.cluster_name
number_of_tests = args.number_of_tests
star_name_of_interest = args.specific_star
baser_dir = args.base_dir
LOG_TO_SCREEN = args.log
split_point = args.split_point
split_size = args.split_size
PLOTTING = args.plotting
DELETE_ALL_FILES = args.overwrite


fitter = fitting.SLSQPLSQFitter()

ca_lines = np.array([8498.03, 8542.09, 8662.14])*u.angstrom
tied_parameters = {'x_0_1': tie_ca_lines_1,
                   'x_0_2': tie_ca_lines_2}
orig_pv = ThreePseudoVoigts(x_0_0=ca_lines[0].value,
                            x_0_1=ca_lines[1].value,
                            x_0_2=ca_lines[2].value,
                            gamma_L_0=2,
                            gamma_G_0=1,
                            amplitude_0=4,
                            gamma_L_1=2,
                            gamma_G_1=1,
                            amplitude_1=4,
                            gamma_L_2=2,
                            gamma_G_2=1,
                            amplitude_2=4,
                            tied=tied_parameters)


base_dir = f"{baser_dir}/{cluster_name}"
phot_table = pd.read_csv(
    "{}/{}_gaia_members.csv".format(
        base_dir, cluster_name), dtype={'NAME': 'str', 'name': 'str'})
phot_table = phot_table.rename(columns={'name': 'NAME'})
if star_name_of_interest is None:
    suffix_string = f"{number_of_tests}"
else:
    suffix_string = f"{star_name_of_interest}"
outfile = outfile_initialize(cluster_name, base_dir, phot_table,
                             suffix=suffix_string)


if split_point == -1:
    table_to_iterate = phot_table
    log_suffix = ""
else:
    table_to_iterate = phot_table.iloc[split_point:split_point+split_size]
    log_suffix = f"_{split_point}"

if LOG_TO_SCREEN:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s')
else:
    logging.basicConfig(
        filename=f'{base_dir}/{cluster_name}{log_suffix}.log',
        filemode='w', level=logging.INFO,
        format='%(asctime)s - %(message)s')

logging.info(f"split{log_suffix}: Starting")
for star_row, star in table_to_iterate.iterrows():
    if star_name_of_interest is None:
        pass
    else:
        if star.NAME not in [star_name_of_interest]:
            continue
    file_location = baser_dir+re.match(f"^[\S]+(/{cluster_name}/[\S]+)",
                                           star.file_name)[1]

    logging.info(f"split{log_suffix}: {star.NAME} {file_location}")
    if DELETE_ALL_FILES:
        if os.path.isfile(pickle_location(file_location)):
            os.remove(pickle_location(file_location))
    try:
        with open(pickle_location(file_location), 'rb') as f:
            star_spectrum = pickle.load(f)
        logging.info(f"split{log_suffix}: {star.NAME}: Already Pickled.")
    except FileNotFoundError:
        star_spectrum = Spectrum(filename=file_location,
                                 star_name=star.NAME,
                                 source_id=star.source_id,
                                 number_of_tests=number_of_tests)
        logging.info(f"split{log_suffix}: {star.NAME}: Not pickled.")
    if star_spectrum.snr < -99.:
        logging.info(f"split{log_suffix}: {star.NAME}: Skipping, low signal.")
        continue
    useful_star_info(outfile, star_spectrum)
    if ((star_spectrum.true_radial_velocity is None) or
        (len(star_spectrum.true_radial_velocity) < number_of_tests)):
        star_spectrum.number_of_tests = number_of_tests
        star_spectrum.initialize_iteration(number_of_tests=number_of_tests)
        star_spectrum.continuum_fitting()

        # Delete the old pickle file if it exists.
        # On Katana is doesn't seem to be overwriting!
        if os.path.isfile(pickle_location(file_location)):
            logging.info(
                f"split{log_suffix}: {star.NAME}: Deleting pickle; needs more iterations.")
            os.remove(pickle_location(file_location))

        logging.info(f"{star.NAME}: Processing")
        for iteration in range(number_of_tests):
            if np.isnan(star_spectrum.fitted_pv_params).all():
                init_pv = orig_pv
            else:
                init_pv = pv_fit
            star_spectrum.continuum_fitting()
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    if number_of_tests > 1:
                        pv_fit = fitter(
                            init_pv,
                            star_spectrum.wavelength_linear.value,
                            (1-star_spectrum.flux_iters[iteration, :] /
                             star_spectrum.cheb_cont[iteration]),
                            verblevel=0)
                    else:
                        pv_fit = fitter(
                            init_pv,
                            star_spectrum.wavelength_linear.value,
                            (1-star_spectrum.flux_iters[iteration] /
                             star_spectrum.cheb_cont[iteration]),
                            verblevel=0)
                except AstropyWarning:
                    logging.info(f"{star.NAME}: AstropyWarning bad fit")
                    continue
            star_spectrum.fitted_pv_params[iteration, :] = pv_fit.parameters
        star_spectrum.save(filename=pickle_location(file_location))
        logging.info(f"split{log_suffix}: {star.NAME}: Finished")
    else:
        logging.info(f"split{log_suffix}: {star.NAME}: Already enough iterations")
    if PLOTTING:
        for iteration in range(number_of_tests):
            star_spectrum.plot_all(iteration=iteration, orig_pv=orig_pv)
    if star_spectrum.ew1 is None:
        outfile.write(",,,")
        outfile.write(",,,")
        outfile.write(",,,")
        outfile.write(",,")
        continue
    percentile_range(star_spectrum.ew1, outfile)
    percentile_range(star_spectrum.ew2, outfile)
    percentile_range(star_spectrum.ew3, outfile)
    percentile_range(star_spectrum.sum_ew, outfile)
    percentile_range(star_spectrum.true_radial_velocity, outfile, rv=True,
                     last=True)
    outfile.write('\n')
outfile.close()
logging.info(f"split{log_suffix}: Finished")
