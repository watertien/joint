"""a Python interface for Fermi-LAT"""
# Energy unit in this file is GeV
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


def model_fit(data, axes, model, optimizer, **kwargs):
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
    kwargs : dict
        parameters to be fitted

    Returns
    -------
    result : fit result object
        Best fit result
    """
    paras_name = ["prefactor", "index", "sigma"]
    # Store unit info
    paras_unit = [kwargs[i].unit for i in paras_name]
    paras_value = [kwargs[i].value for i in paras_name]
    # Remove unit from kwarg for fit convenience
    kwargs.update(dict(zip(paras_name, paras_value)))
    x0 = kwargs.pop("x0")

    def model_likelihood(paras_value):
        """Objective function be to minizied for model fitting

        Parameters
        ----------
        paras_values : (n,) ndarray
            array containing parameter to be fitted

        Return
        ------
        nlog_likelihood : float
            negative of log likelihood for parameter set of given model
        """
        # TODO : replace constant list of parameter name with more general method
        paras_dict = dict(zip(paras_name, paras_value * paras_unit))
        # Update kwargs when fitting
        kwargs.update(paras_dict)
        pred_cube = get_pred_cube(axes, model, **kwargs)
        # Apply psf if True
        if kwargs["psf"]:
            pred_cube = apply_psf(axes, pred_cube, kwargs["psf_cube"])
        return -np.sum(get_likelihood(pred_cube, data))

    from scipy.optimize import minimize
    
    result = minimize(model_likelihood, x0,\
                       method=optimizer)
    # Restore unit info
    result.x = np.array(result.x * paras_unit)
    return result


def crab_skymodel(axes, prefactor, index, sigma, **kwargs):
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
    prefactor : astropy.quantity
        prefactor for spectral model Unit: count/GeV/cm2/s
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
            dN/dE value at given energy Unit: count/GeV/cm/s
        """
        referennce_energy = 1 * u.GeV
        return prefactor * (energy / referennce_energy)**index

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
    # TODO : consider integrating dN/dE
    # Use a temporary fake integral here 
    energy_binsize = axes[0][1] - axes[0][0] # energy unit
    # background is count(dimensionless)
    return model(axes, **kwargs) * exposure * energy_binsize + background


def get_ts_cube(model, data, axes, exposure, background, **kwargs):
    """Calculte test statistic cube for a model
    
    Parameters
    ----------
    model : function
        model to be test
    data : 3-D array
        observation data
    axes : tuple of 1-D array
        cube coordinate
    exposure : 3-D array
        exposure cube
    background : 3-D array
        background cube
    
    Returns
    -------
    ts_cube : 3-D array
        test statistic cube for given model and data
    """
    bkg_like = get_likelihood(background, data)
    pred_cube = get_pred_cube(axes, model, exposure, background, **kwargs)
    model_like = get_likelihood(pred_cube, data)
    ts_cube = -2 * (bkg_like - model_like)
    return ts_cube


def get_likelihood(pred_cube, counts):
    """Calculate binned likelihood
    
    Parameters
    ----------
    pred_cube : 3-D array
        prediction cube from model
    counts : 3-D array
        observation data cube

    Returns
    -------
    log_like : 3-D array
        log likelihood of parameters for prediction cube
    """
    # In order to avoid too large number for fatorial
    # use Stirling's Approximation here, which is
    # ln(n!) = n * ln(n) - n (first order approximation)
    mask_large = counts > 10
    counts_masked = np.ma.masked_array(counts, mask=mask_large)
    # Only elements less than 10 is valid
    # TODO : use masked array to do condition slicing
    # counts_masked = np.ma.masked_array(counts, mask=mask_large)
    counts_masked = counts.copy()
    counts_factorial = np.zeros_like(counts)
    counts_factorial[~mask_large] = np.log10(factorial(counts_masked[~mask_large]))
    counts_factorial[mask_large] = counts_masked[mask_large] * np.log10(counts_masked[mask_large]) \
                                            - counts_masked[mask_large] * np.log10(np.e)
    # For a given bin range, counts_factorial is reusable
    # TODO : What if element in pred_cube equals zero?
    # To avoid zero division, add a tiny quantity to pred_cube
    log_like = counts * np.log10(pred_cube + 1e-12) \
                - pred_cube * np.log10(np.e) - counts_factorial
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
        pos = ax.imshow(_map, cmap="coolwarm")
        fig.colorbar(pos, ax=ax)
    # Plot SED for center pixel
    axs[-1].plot(cube[:, 20, 25], "ro--")
    axs[-1].loglog()
    fig.tight_layout()
    return fig, axs


def apply_psf(axes, pred_cube, psf_cube):
    """Apply Point Spread Function(PSF) to prediction cube

    Parameters
    ----------
    axes : tuple of 1-D arrays
        (energy, dx, dy) axes in binned likelihood analysis
    pred_cube : 3-D array
        prediction cube given by model
    psf_cube : 3-D array
        PSF cube

    Returns
    -------
    pred_with_psf : 3-D array
        2-D(spatial direction) convolution of pred_cube and psf_cube 
    """
    # TODO : support binning in this function
    # Gammapy uses scipy.signal.fftconvolve  
    from scipy.signal import convolve2d
    if pred_cube.ndim != psf_cube.ndim:
        raise ValueError("Input arrays must have the same dimension\n"
                        + f"(pred_cube{pred_cube.shape}, psf_cube{psf_cube.shape}")
    # Make sure the first dimension is ENERGY
    pred_with_psf = np.zeros_like(pred_cube)
    for i, map2d in enumerate(pred_cube):
        # TODO : determine kwargs for convolve2d
        # map2d should be counts without any unit
        pred_with_psf[i] = convolve2d(map2d, psf_cube[i], mode="same")
    return pred_with_psf
