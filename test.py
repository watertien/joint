import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from joint import *


"""
def hdu_plot():
       guess_pred = get_pred_cube((energy_center, dx, dy), crab_skymodel,\
                                   exposure, background, **guess_parameter)
       guess_likelihood = get_likelihood(guess_pred, counts)
       print(f"data: {get_likelihood(counts, counts):.3E}\n"
              f"guess: {guess_likelihood:.3E}")
       dx = np.arange()
       dy = np.arange()
       dx = np.linspace(-1, 1, 50) * u.degree
       dy = np.linspace(-1, 1, 40) * u.degree
       axes = (energy_center, dx, dy)
       pred_cube = crab_skymodel(axes, **test_parameter)
       test_slice = slice(5)

       fig, axs = plot_cube(pred_cube, test_slice)
       axs[-1].lines[0].set_xdata(energy_center)
       axs[-1].set_xlim(energy_lo[0].value, energy_hi[-1].value)
       axs[-1].set_xlabel("TeV")
       fig.savefig(f"test_pred_cube_σ{sigma}.png", dpi=500)
       fig1, axs1 = plot_cube(background, test_slice)
       axs1[-1].lines[0].set_xdata(energy_center)
       axs1[-1].set_xlim(energy_lo[0].value, energy_hi[-1].value)
       axs1[-1].set_xlabel("TeV")
       fig1.savefig(f"test_bkg_cube_σ{sigma}.png", dpi=500)
       fig.savefig(f"test_pred_cube_σ{test_parameter["sigma"]}.png", dpi=500)
"""


def test_fit():
    fname = "/home/tian/Documents/my_joint/Fermi-LAT-3FHL_data_Fermi-LAT.fits"
    fermi_hdul = read(fname)

    # Log interpolation, center = (lo * hi)^0.5
    energy_center = fermi_hdul["COUNTS_BANDS"].data.field(1) * u.TeV
    energy_lo = fermi_hdul["COUNTS_BANDS"].data.field(2) * u.TeV
    energy_hi = fermi_hdul["COUNTS_BANDS"].data.field(3) * u.TeV
    # BANDS are the same for COUNTS, BKG, PSF, so just assign once here
    exposure = fermi_hdul["EXPOSURE"].data * u.Unit(fermi_hdul["EXPOSURE"].header["BUNIT"])
    background = fermi_hdul["BACKGROUND"].data * u.Unit("")
    counts = fermi_hdul["COUNTS"].data * u.Unit("")
    psf_cube = fermi_hdul["PSF_KERNEL"].data * u.Unit("")

    # Read deltx and delty from FITS 
    deltx = fermi_hdul["COUNTS"].header["CDELT1"] 
    delty = fermi_hdul["COUNTS"].header["CDELT2"] 
    dx_edges = np.linspace(-25, 25, 51) * u.degree * deltx
    dy_edges = np.linspace(-20, 20, 41) * u.degree * delty
    # Use averge value for position center coordinate
    dx = (dx_edges[:-1] + dx_edges[1:]) / 2
    dy = (dy_edges[:-1] + dy_edges[1:]) / 2

    guess_parameter = {"prefactor": 1e-10 * u.Unit("cm-2 s-1 GeV-1"), \
                     "index": -3.18 * u.Unit(""),\
                     "sigma": 0.1 * u.degree}
    
    fit_kwargs = {"x0": [1e-10, -3.18, 0.2]}
    fit_kwargs.update(guess_parameter)
    fit_kwargs["exposure"] = exposure
    fit_kwargs["background"] = background
    fit_kwargs["psf"] = False
    fit_kwargs["psf_cube"] = psf_cube
    axes = (energy_center, dx, dy)

    test_fit_result = model_fit(counts, axes, crab_skymodel, "Nelder-Mead", **fit_kwargs)
    print(test_fit.x, test_fit.message, sep='\n')
    return test_fit_result


if __name__ == "__main__":
    test_fit()
