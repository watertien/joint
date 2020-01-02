import astropy.units as u
import numpy as np
from astropy.visualization import quantity_support
quantity_support()
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
    guess_parameter = {"prefactor": 1e-10 * u.Unit("cm-2 s-1 GeV-1"), \
                     "index": -3.18 * u.Unit(""),\
                     "sigma": 0.1 * u.degree}
    
    fit_kwargs = {"x0": [1e-10, -3.18, 0.2]}
    fit_kwargs["paras_name"] = {"prefactor", "index", "sigma"}
    fit_kwargs.update(guess_parameter)
    fit_kwargs["exposure"] = exposure
    fit_kwargs["background"] = background
    fit_kwargs["psf"] = False
    fit_kwargs["psf_cube"] = psf_cube
    axes = (energy_center, dx, dy)
    test_fit_result = model_fit(counts, axes, crab_skymodel, "Nelder-Mead", **fit_kwargs)
    print(test_fit_result.x, test_fit_result.message, sep='\n')
    return test_fit_result


def test_sensitivity():
    axes = (energy_center, dx, dy)
    diff_sen = get_sensitivity(crab_skymodel, counts, axes, exposure, background)
    guess_parameter = {"prefactor": 1e-10 * u.Unit("cm-2 s-1 GeV-1"), \
                     "index": -2 * u.Unit(""),\
                     "sigma": 0.1 * u.degree}
    ts_cube = np.zeros_like(counts)

    for i, sen in enumerate(diff_sen):
        # Calculate TS on each energy bin
        guess_parameter.update({"prefactor": sen})
        sub_axes = (energy_center[i, np.newaxis], dx, dy)
        ts_cube[i] = get_ts_cube(crab_skymodel, counts[i, np.newaxis], sub_axes,\
                                exposure[i, np.newaxis], background[i, np.newaxis],
                                **guess_parameter)
    return ts_cube


def test_sens_ts():
    # axes = (energy_center, dx, dy)
    # Calculate TS for different prefactors
    test_sens = np.logspace(-12, -7, num=10) * u.Unit("cm-2 s-1 GeV-1")
    ts_prefactor = np.zeros(test_sens.size)
    guess_parameter = {"prefactor": 1e-10 * u.Unit("cm-2 s-1 GeV-1"), \
                     "index": -2 * u.Unit(""),\
                     "sigma": 0.1 * u.degree}
    fig, axs = plt.subplots(nrows=energy_center.size, ncols=1,\
                            sharex=True, figsize=(10, 20))
    for i in range(energy_center.size):
        sub_axes = (energy_center[i, np.newaxis], dx, dy)
        for j, sen in enumerate(test_sens):
            # Calculate TS on each energy bin
            guess_parameter.update({"prefactor": sen})
            ts_cube = get_ts_cube(crab_skymodel, counts[i, np.newaxis], sub_axes,\
                                    exposure[i, np.newaxis], background[i, np.newaxis],
                                    **guess_parameter)
            ts_prefactor[j] = np.sum(ts_cube)
        axs[i].plot(test_sens, ts_prefactor, 'o--',\
                    label=f"energy index{i}, {energy_center[i]:.3f}")
        axs[i].axhline(y=25, lw=3, color='red')
        axs[i].legend()
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.xscale("log")
    plt.savefig("ts_for_different_prefactor.png", dpi=500)
    return test_sens, ts_prefactor


fname = "Fermi-LAT-3FHL_data_Fermi-LAT.fits"
fermi_hdul = read(fname)

# Log interpolation, center = (lo * hi)^0.5
energy_center = fermi_hdul["COUNTS_BANDS"].data.field(1) * u.TeV
energy_lo = fermi_hdul["COUNTS_BANDS"].data.field(2) * u.TeV
energy_hi = fermi_hdul["COUNTS_BANDS"].data.field(3) * u.TeV
# BANDS are the same for COUNTS, BKG, PSF, so just assign once here
exposure = fermi_hdul["EXPOSURE"].data * u.Unit(fermi_hdul["EXPOSURE"].header["BUNIT"])
background = fermi_hdul["BACKGROUND"].data * u.Unit("")
counts = fermi_hdul["COUNTS"].data * u.Unit("")
counts1 = counts.copy()
# counts1[-1, 20:30, 20:30] += 1000
psf_cube = fermi_hdul["PSF_KERNEL"].data * u.Unit("")

# Read deltx and delty from FITS 
deltx = fermi_hdul["COUNTS"].header["CDELT1"] 
delty = fermi_hdul["COUNTS"].header["CDELT2"] 
dx_edges = np.linspace(-25, 25, 51) * u.degree * deltx
dy_edges = np.linspace(-20, 20, 41) * u.degree * delty
# Use averge value for position center coordinate
dx = (dx_edges[:-1] + dx_edges[1:]) / 2
dy = (dy_edges[:-1] + dy_edges[1:]) / 2

if __name__ == "__main__":
    # ts_cube = test_sensitivity()
    # ts_arr = np.sum(ts_cube, axis=(1, 2))
    # Test impact of prefactor for TS
    sens, ts = test_sens_ts()
    # fit_dict = dict(zip(["prefactor", "index", "sigma"], fit_result.x))
    # fit_pred = get_pred_cube(axes, crab_skymodel, exposure, background, **fit_dict)
    # ts_cube = get_ts_cube(crab_skymodel, counts, axes, exposure, background, **fit_dict)
    # fig, axs = plot_cube(ts_cube, slice(5))
    # fig.savefig("test_ts_map.png", dpi=500)
    # plt.show()
