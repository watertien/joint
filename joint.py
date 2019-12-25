"""a Python interface for Fermi-LAT"""
from astropy.io import fits
import astropy.units as u
import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt


def read(fname):
    """read Fermi-LAT fits file(pre-made by gammapy)

    Parameters
    ----------
    fname : str
         Filename of data file

    Returns
    -------
    out : astropy.io.fits.HDUList
        HDUList object of input fits file

    Examples
    --------
    >>> hdul = read("Fermi-LAT-3FHL_data_Fermi-LAT.fits")
    >>> print(hdul.info)
    'Fermi-LAT-3FHL_data_Fermi-LAT.fits'
    """

    return fits.open(fname, mode="readonly")


def get_flux(hdulist, axes):
    """Calculate flux points from raw count data

    Parameters
    ----------
    hdulist : HDUList object
        HDUList containing COUNTS, BACKGROUND, E_DISP, PSF HDUs
    axes : tuple of 1-D array
        Coordinate of cube point, in the order of counts data
        ie. (ENERGY, LON, LAT) respectively
    Returns
    -------
    flux : NDarray
        flux cube for original count data
    """


def model_fit(data, axes, model, optimizer):
    """Fit data with given model

    Parameters
    ----------
    data : NDarray
        Flux data cube
    axes : tuple of 1-D array
        Coordinate of cube point
    model : function
        Function that can calculate flux for given coordinate
    optimizer : str
        Optimization method to be used in scipy.optimize,
        options are: BFGS, COBYLA, least_squares etc.

    Returns
    -------
    fit_parameters : 1-D array
        Best fit parameter set
    fit_covariance : matrix
        Covariance matrix of fit result
    """


def crab_skymodel(axes, prefactor, index, sigma):
    """Spatial and spectral model of crab

    Parameters
    ----------
        energy : ndarray
            energy coordinates
        dely : ndarray
            y offset
        delx : ndarray
            x offset
    axes : tuple of 1-D array
        cube axes
    prefactor : float
        prefactor for spectral model
    index : float
        power law index
    sigma : angle
        spatial sigma

    Returns
    -------
    crab : ndarray
        flux of crab nebula in sky for power-law
    """
    energy, delx, dely = axes

    def crab_spatial(delx, dely):
        """Spatial model of crab, 2-D Gaussian distribution

        Parameters
        ----------
        delx : angle
            offset angle in x direction
        dely : angle
            offset angle in y direction
       
        Returns
        -------
        out : float
            spatial value
        """
        return np.exp(-(delx**2 + dely**2) / (2 * sigma**2))

    def crab_spctral(energy):
        """Spectal model of crab, power-law

        Parameters
        ----------
        energy : or astropy.Quantity
            energy points at which to calculate
        
        Returns
        -------
        out : astropy.Quantity
            spectral value
        """
        refernce_energy = 100 * u.GeV
        return prefactor * (energy / refernce_energy)**index

    # TODO: order of meshgrid is not clear
    yy, ee, xx, = np.meshgrid(dely, energy, delx)
    return crab_spatial(xx, yy) * crab_spctral(ee)


def get_pred_cube(axes, model, exposure, background, **kwargs):
    """Calculate model prediction cube

    Parameters
    ----------
    axes : tuple of 1-D array
        energy, dx, dy center of bins
    model : function
        source skymodel
    exposure : 3-D array
        exposure for each data bin
    background : 3-D array
        background data cube
    **kwargs : dict
        parameter set of input model
    
    Returns
    -------
    pred_cube : 3-D array
        prediction data cube of given model and background
    """
    return model(axes, **kwargs) * exposure + background


def get_likelihood(pred_cube, counts, **kwargs):
    """Calculate binned likelihood
    
    Parameters
    ----------
    pred_cube : 3-D array
        prediction cube from model
    counts : 3-D array
        observation data cube

    Returns
    -------
    out : float
        log likelihood of parameters for prediction cube
    """
    pred_total = np.sum(pred_cube)
    # In order to avoid too large number for fatorial
    # use Stirling's Approximation here, which is
    # ln(n!) = n * ln(n) - n (first order approximation)
    mask_large = counts > 10
    counts_masked = np.ma.masked_array(counts, mask=mask_large)
    mask_large = counts > 10
    # Only elements less than 10 is valid
    # TODO : use masked array to do condition slicing
    # counts_masked = np.ma.masked_array(counts, mask=mask_large)
    counts_masked = counts.copy()
    counts_factorial = np.zeros_like(counts)
    counts_factorial[~mask_large] = np.log10(factorial(counts_masked[~mask_large]))
    counts_factorial[mask_large] = counts_masked[mask_large] * np.log10(counts_masked[mask_large]) \
                                            - counts_masked[mask_large] * np.log10(np.e)
    # For a given bin range, data_fatorial is reusable
    data_factorial = np.sum(counts_factorial)
    # TODO : What if element in pred_cube equals zero?
    # To avoid zero division, add a tiny quantity to pred_cube
    log_like = np.sum(counts * np.log10(pred_cube + 1e-12)) - data_factorial
    log_like = -pred_total * np.log10(np.e) + log_like
    return log_like


def plot_cube(cube, index):
    """Plot data cube for convenience

    Parameters
    ----------
    cube : 3-D array
        Input data
    index : slice object
        index of maps to be plotted

    Returns
    -------
    fig : figure object
    axs : list of axes object
    """
    _maps = cube[index]
    fig, axs = plt.subplots(nrows=len(_maps) + 1, figsize=(10, 20))
    # Plot 2D map for every energy bin
    for ax, _map in zip(axs, _maps):
        ax.imshow(_map)
    # Plot SED for center pixel
    axs[-1].plot(cube[:, 20, 25], "ro--")
    axs[-1].loglog()
    fig.tight_layout()
    return fig, axs


if __name__ == "__main__":
    fname = "./Fermi-LAT-3FHL_data_Fermi-LAT.fits"
    fermi_hdul = read(fname)

    # Log interpolation, center = (lo * hi)^0.5
    energy_center = fermi_hdul["COUNTS_BANDS"].data.field(1) * u.TeV
    energy_lo = fermi_hdul["COUNTS_BANDS"].data.field(2) * u.TeV
    energy_hi = fermi_hdul["COUNTS_BANDS"].data.field(3) * u.TeV
    # BANDS are the same for COUNTS, BKG, PSF, so just assign once here
    exposure = fermi_hdul["EXPOSURE"].data
    background = fermi_hdul["BACKGROUND"].data
    counts = fermi_hdul["COUNTS"].data
