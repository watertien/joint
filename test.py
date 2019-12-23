from joint import read, crab_skymodel, plot_cube
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    fname = "/home/tian/Documents/my_joint/Fermi-LAT-3FHL_data_Fermi-LAT.fits"
    fermi_hdul = read(fname)

    # Log interpolation, center = (lo * hi)^0.5
    energy_center = fermi_hdul["COUNTS_BANDS"].data.field(1) * u.TeV
    energy_lo = fermi_hdul["COUNTS_BANDS"].data.field(2) * u.TeV
    energy_hi = fermi_hdul["COUNTS_BANDS"].data.field(3) * u.TeV
    # BANDS are the same for COUNTS, BKG, PSF, so just assign once here
    exposure = fermi_hdul["EXPOSURE"].data
    background = fermi_hdul["BACKGROUND"].data
    counts = fermi_hdul["COUNTS"].data

    test_parameter = {"index": -3, "sigma": 0.3 * u.degree}
    sigma = test_parameter["sigma"]
    # dx = np.arange()  
    # dy = np.arange()  
    dx = np.linspace(-1, 1, 50) * u.degree
    dy = np.linspace(-1, 1, 40) * u.degree
    axes = (energy_center, dx, dy)
    pred_cube = crab_skymodel(axes, **test_parameter)
    test_slice = slice(5) 
    # fig, axs = plot_cube(pred_cube, test_slice)
    # axs[-1].lines[0].set_xdata(energy_center)
    # axs[-1].set_xlim(energy_lo[0].value, energy_hi[-1].value)
    # axs[-1].set_xlabel("TeV")
    # fig.savefig(f"test_pred_cube_σ{sigma}.png", dpi=500)
    fig1, axs1 = plot_cube(background, test_slice)
    axs1[-1].lines[0].set_xdata(energy_center)
    axs1[-1].set_xlim(energy_lo[0].value, energy_hi[-1].value)
    axs1[-1].set_xlabel("TeV")
    fig1.savefig(f"test_bkg_cube_σ{sigma}.png", dpi=500)
    
    # fig.savefig(f"test_pred_cube_σ{test_parameter["sigma"]}.png", dpi=500)
