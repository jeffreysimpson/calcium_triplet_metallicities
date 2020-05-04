# Empirical relationship between calcium triplet equivalent widths and [Fe/H]  using Gaia photometry

I present a new empirical relationship for red giant branch stars between the overall metallicity of the star and the sum of equivalent widths of the near-infrared calcium triplet (CaT) spectral lines. This method takes advantage of the all-sky photometry and astrometry of the Gaia mission, and the archival spectra from 2050 red giant branch stars from 18 globular clusters (-0.69>[Fe/H]>-2.44) acquired with the Anglo-Australian Telescope's AAOmega spectrograph.

## Installation

```bash
git clone https://github.com/jeffreysimpson/calcium_triplet_metallicities.git
```

requirements.txt contains the particular versions of external packages that I used.

## Usage

```bash
python running.py -n 100 -d ../data/spectra --log_to_screen -c NGC7078
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
