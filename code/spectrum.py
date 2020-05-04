#!/usr/bin/env python

"""Spectrum class for AAOmega spectra."""

import os
import pickle
# import re

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time

from PseudoVoigt import ThreePseudoVoigts, tie_ca_lines_1, tie_ca_lines_2

__author__ = "Jeffrey Simpson"
__copyright__ = "Copyright 2020, Jeffrey Simpson"
__credits__ = ["Jeffrey Simpson"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jeffrey Simpson"
__email__ = "jeffrey.simpson@unsw.edu.au"
__status__ = "Development"


mask_size = np.array([10, 10, 10]) * u.angstrom
ca_lines = np.array([8498.03, 8542.09, 8662.14]) * u.angstrom
cont_regions = np.array([[8474, 8484],
                         [8563, 8577],
                         [8619, 8642],
                         [8700, 8725],
                         [8776, 8792]]) * u.angstrom
tied_parameters = {'x_0_1': tie_ca_lines_1,
                   'x_0_2': tie_ca_lines_2}


def assure_path_exists(path):
    """Check save directory exists, and if not, create it."""
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def initialize_pv(parameters):
    """Create an arbitrary ThreePseudoVoigts."""
    return ThreePseudoVoigts(x_0_0=parameters[0],
                             gamma_L_0=parameters[1],
                             gamma_G_0=parameters[2],
                             amplitude_0=parameters[3],
                             x_0_1=parameters[4],
                             gamma_L_1=parameters[5],
                             gamma_G_1=parameters[6],
                             amplitude_1=parameters[7],
                             x_0_2=parameters[8],
                             gamma_L_2=parameters[9],
                             gamma_G_2=parameters[10],
                             amplitude_2=parameters[11],
                             tied=tied_parameters)


def empty_axes_test(axes):
    """Create an empty axes if needed."""
    if axes is None:
        fig, axes = plt.subplots(ncols=1,
                                 nrows=1,
                                 figsize=(5, 10))
        return axes
    else:
        return axes


class Spectrum(object):
    """Spectrum read from AAOmega FITS file."""

    def __init__(self,
                 filename=None,
                 star_name=None,
                 rv_guess=None,
                 source_id=None,
                 number_of_tests=1):
        """Initialize stuff."""
        if filename is None:
            raise FileNotFoundError("FITS file not specified")
        if star_name is None:
            raise ValueError("Star name required.")
        self.filename = filename
        self.star_name = star_name
        self.source_id = source_id
        self.number_of_tests = number_of_tests
        self.get_star_info()
        self.initialize_iteration(number_of_tests=number_of_tests)
        self.continuum_fitting()
        self._measured_radial_velocity = rv_guess

    @property
    def rv_offset(self):
        """Calcuate the multiplicative factor for wavelength space."""
        if np.isnan(self.fitted_pv_params).all():
            self._rv_offset = 1.
        else:
            self._rv_offset = (
                1/(np.nanmedian(self.measured_radial_velocity)/const.c + 1))
        return self._rv_offset

    @property
    def measured_radial_velocity(self):
        """Calcute measured radial velocity."""
        if np.isnan(self.fitted_pv_params).all():
            self._measured_radial_velocity = None
        else:
            line1_list = self.fitted_pv_params[:, 0] * u.angstrom
            self._measured_radial_velocity = (
                ((line1_list - ca_lines[0]) / (ca_lines[0])) * const.c)
        return self._measured_radial_velocity

    @property
    def true_radial_velocity(self):
        """Calcute true radial velocity using the barycentric correction."""
        if np.isnan(self.fitted_pv_params).all():
            return None
        else:
            return (self.measured_radial_velocity +
                    self.barycorr +
                    self.measured_radial_velocity*self.barycorr/const.c)

    @property
    def ew1(self):
        """Get the array of ew1s."""
        if np.isnan(self.fitted_pv_params).all():
            return None
        else:
            return self.fitted_pv_params[:, 3]

    @property
    def ew2(self):
        """Get the array of ew2s."""
        if np.isnan(self.fitted_pv_params).all():
            return None
        else:
            return self.fitted_pv_params[:, 7]

    @property
    def ew3(self):
        """Get the array of ew3s."""
        if np.isnan(self.fitted_pv_params).all():
            return None
        else:
            return self.fitted_pv_params[:, 11]

    @property
    def sum_ew(self):
        """Calcute the sum of the three CaT lines."""
        if np.isnan(self.fitted_pv_params).all():
            self._sum_ew = None
        else:
            self._sum_ew = self.ew1 + self.ew2 + self.ew3
        return self._sum_ew

    def get_star_info(self):
        """Get useful information from the file.

        This finds the fibre number, wavelength, flux, barycentric correction.
        The flux and wavelength arrays returned only have non-NaNs values.
        """
        spec_dir = os.path.dirname(self.filename)
        self.arc_lamp = ""
        for file in os.listdir(spec_dir):
            if file.endswith(".fits"):
                open_fits = fits.open(f"{spec_dir}/{file}")
                if open_fits[0].header['RUNCMD'] == "ARC":
                    self.arc_lamp = open_fits[0].header['LAMPNAME']
                    break
        open_fits.close()

        hdul = fits.open(self.filename)
        fibre_names = hdul[2].data['NAME']
        self.fibre_num = np.arange(len(fibre_names))[fibre_names == self.star_name][0]
        fibre_idx = np.arange(len(fibre_names)) == self.fibre_num

        # Remove all the NaNs.
        flux_raw = hdul[0].data[fibre_idx][0]
        no_nans_idx = (~np.isnan(flux_raw))
        self.wavelength_linear = (
            (hdul[0].header['CDELT1'] *
             (np.arange(2048)+1-hdul[0].header['CRPIX1']) +
             hdul[0].header['CRVAL1']) * u.angstrom)[no_nans_idx]
        self.flux = hdul[0].data[fibre_idx][0][no_nans_idx]
        self.variance = hdul[1].data[fibre_idx][0][no_nans_idx]
        self.header = hdul[0].header

        self.star_position = SkyCoord(
            ra=hdul[2].data['RA'][fibre_idx][0]*u.rad,
            dec=hdul[2].data['DEC'][fibre_idx][0]*u.rad,
            frame='icrs')
        self.obtime = Time(hdul[0].header['UTMJD'], format='mjd', scale='utc')
        self.telescope_location = EarthLocation.from_geodetic(
            lat=hdul[0].header['LAT_OBS']*u.deg,
            lon=hdul[0].header['LONG_OBS']*u.deg,
            height=hdul[0].header['ALT_OBS']*u.m)
        self.barycorr = self.star_position.radial_velocity_correction(
            obstime=self.obtime, location=self.telescope_location)
        self.snr = np.nanmedian(self.flux / np.sqrt(self.variance))
        hdul.close()

    def initialize_iteration(self, number_of_tests=1):
        """Initialize the spectrum for the iteration."""
        assert number_of_tests > 0, "Give a positive number of tests."
        assert isinstance(number_of_tests, int)
        if number_of_tests > 1:
            random_error = np.random.randn(
                number_of_tests, len(self.wavelength_linear))
            self.flux_iters = (self.flux + random_error *
                               np.sqrt(self.variance))
        else:
            self.flux_iters = np.array([self.flux])
        self.fitted_pv_params = np.zeros((number_of_tests, 12))
        self.fitted_pv_params[:] = np.nan

    def continuum_fitting(self):
        """Do the continuum fitting."""
        # Masks out each of the CaT lines
        cont_idx = np.ones(len(self.wavelength_linear), dtype=np.bool)

        for idx, ca_line in enumerate(ca_lines):
            ca_mask_1 = (self.wavelength_linear*self.rv_offset >=
                         ca_line-mask_size[idx])
            ca_mask_2 = (self.wavelength_linear*self.rv_offset <=
                         ca_line+mask_size[idx])
            cont_idx[np.where(np.logical_and(ca_mask_1,
                                             ca_mask_2))] = False
        self.cont_idx = cont_idx

        # Fit a fifth order Chebychev to get the continuum.
        # This is done for every possible spectrum simultaneously.
        self.fitted_cheb_coeffs = np.polynomial.legendre.legfit(
            self.wavelength_linear[self.cont_idx],
            self.flux_iters[:, self.cont_idx].T, 4)
        self.cheb_cont = np.polynomial.legendre.legval(
            self.wavelength_linear.value,
            self.fitted_cheb_coeffs)

        mean_vals = np.zeros((5, self.number_of_tests))
        mean_waves = np.zeros(5)
        mean_vals[:] = np.nan
        mean_waves[:] = np.nan

        for count_count, cont_region in enumerate(cont_regions):
            cont_mask_1 = (self.wavelength_linear*self.rv_offset >=
                           cont_region[0])
            cont_mask_2 = (self.wavelength_linear*self.rv_offset <=
                           cont_region[1])
            cont_region_idx = np.where(np.logical_and(cont_mask_1,
                                                      cont_mask_2))
            mean_waves[count_count] = np.mean(
                self.wavelength_linear[cont_region_idx]).value
            mean_vals[count_count] = np.nanmean(
                (self.flux_iters/self.cheb_cont)[
                    :, cont_region_idx][:, 0, :], axis=1)
        try:
            self.cont_fit_adjust_coeffs = np.polynomial.polynomial.polyfit(
                x=mean_waves, y=mean_vals, deg=1)
        except ValueError:
            self.cont_fit_adjust_coeffs = np.array([np.nan, np.nan])
        self.cont_fit_adjust = np.polynomial.polynomial.polyval(
            self.wavelength_linear.value, self.cont_fit_adjust_coeffs)
        self.cheb_cont *= self.cont_fit_adjust

    def plot_orig(self, axes=None):
        """Create a plot of the spectrum."""
        axes = empty_axes_test(axes)
        axes.plot(self.wavelength_linear, self.flux/np.median(self.flux))

    def plot_an_iter_flux(self, iter_num=None, axes=None):
        """Create a plot of the spectrum."""
        assert iter_num is not None, "Give a positive iter_num."
        assert isinstance(iter_num, int), "Give an integer."
        axes = empty_axes_test(axes)
        axes.plot(
            self.wavelength_linear,
            (self.flux_iters[iter_num, :] /
             np.median(self.cheb_cont[iter_num])))

    def plot_a_cheb_cont(self, iter_num=None, axes=None):
        """Create a plot of the spectrum."""
        assert iter_num is not None, "Give a positive iter_num."
        assert isinstance(iter_num, int), "Give an integer."
        axes = empty_axes_test(axes)
        axes.plot(self.wavelength_linear,
                  self.cheb_cont[iter_num]/np.median(self.cheb_cont[iter_num]))

    def plot_a_normed_spec(self, iter_num=None, axes=None):
        """Create a plot of the spectrum."""
        assert iter_num is not None, "Give a positive iter_num."
        assert isinstance(iter_num, int), "Give an integer."
        axes = empty_axes_test(axes)
        axes.plot(self.wavelength_linear,
                  self.flux_iters[iter_num, :]/self.cheb_cont[iter_num])

    def plot_pseudovoigt(self, iter_num=None, axes=None):
        """Plot the fitted PseudoVoigt1D."""
        assert iter_num is not None, "Give a positive iter_num."
        assert isinstance(iter_num, int), "Give an integer."
        axes = empty_axes_test(axes)
        axes.plot(
            self.wavelength_linear,
            1-initialize_pv(
                self.fitted_pv_params[iter_num, :])(
                    self.wavelength_linear.value))

    def plot_all(self, iteration, filename=None, orig_pv=None):
        """Convenvience function to plot everything."""
        if filename is None:
            # save_dir = os.path.dirname(self.filename) + "/spec_plots/"
            save_dir = "spec_plots/"
            base_filename = f"{save_dir}{self.star_name}"
            assure_path_exists(base_filename)
        file_count = 0
        while os.path.exists(f"{base_filename}_{file_count}.pdf"):
            file_count += 1
        plot_filename = f"{base_filename}_{file_count}.pdf"

        fig, axes = plt.subplots(ncols=1,
                                 nrows=2,
                                 figsize=(3.32*6, 10))
        self.plot_an_iter_flux(iter_num=iteration, axes=axes[0])
        self.plot_a_cheb_cont(iter_num=iteration, axes=axes[0])
        self.plot_a_normed_spec(iter_num=iteration, axes=axes[1])
        self.plot_pseudovoigt(iter_num=iteration, axes=axes[1])
        if orig_pv is not None:
            axes[1].plot(self.wavelength_linear,
                         1-orig_pv(self.wavelength_linear.value))
        axes[0].set_ylim(-0.1, 2.0)
        axes[1].set_ylim(-0.1, 1.2)
        axes[0].set_title(f"{self.filename} {self.star_name} {self.source_id}")
        plt.savefig(f'{plot_filename}', bbox_inches="tight")
        plt.close('all')

    def save(self, filename=None):
        """Pickle the spectrum."""
        if filename is None:
            save_dir = os.path.dirname(self.filename) + "/pickles/"
            filename = f"{save_dir}{self.star_name}.pickle"
        assure_path_exists(filename)
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
