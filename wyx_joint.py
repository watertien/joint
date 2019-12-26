import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import factorial 


class BaseSpatialModel:
    '''
    Base class for geometry. 
    '''
    
    def __init__(self,skydir,width,binsz):
        self.skydir=skydir
        self.width=width
        self.binsz=binsz

        self.x_edge=np.arange(0,width[0]+binsz,binsz)-width[0]/2+skydir[0]
        self.y_edge=np.arange(0,width[1]+binsz,binsz)-width[1]/2+skydir[1]
        
        self.x_center=(self.x_edge[:-1]+self.x_edge[1:])/2
        self.y_center=(self.y_edge[:-1]+self.y_edge[1:])/2
        
class GaussianSpatialModel(BaseSpatialModel):
    '''
    Gaussian geometric Model
    '''
    
    def __init__(self,sigma,*args):
        self.sigma=sigma
        super().__init__(*args)

    @property
    def spatial(self):
        xx,yy=np.meshgrid(self.x_center,self.y_center)
        return np.exp(-(xx**2 + yy**2) / (2 * self.sigma**2)).T
    
    
class Axes:
    '''
    Base class for axes
    '''
    
    def __init__(self,name=None):
        self.name=name
        self.edges=None
        self.centers=None
    
    def make_edges(self,width=(-2,2),interp='log',num=10):
        if interp=='log':
            edges=np.linspace(width[0],width[1],num)
            centers=(edges[1:]+edges[:-1])/2
            
            self.edges=np.power(10,edges)
            self.centers=np.power(10,centers)
            
    def from_edges(self,edges):
        self.edges=edges
        
        # Calculate the centers.
        edges=np.log10(edges)
        centers=(edges[:-1]+edges[1:])/2
        self.centers=np.power(10,centers)
        
        
class PowerLawSpctralModel(Axes):
    '''
    
    '''
    
    def __init__(self,prefactor,index,refernce_energy=100,*args,**kwargs):
        self.prefactor=prefactor
        self.index=index
        self.refernce_energy=refernce_energy
        super().__init__(*args,**kwargs)
        #self.parameters={'prefactor':prefactor,'index':index,'sigma':sigma}
        
    @property
    def spectrum(self):
        return self.prefactor * (self.centers / self.refernce_energy)**self.index
        
        
class MapModel:
    
    def __init__(self,geom=None,axes=None,exposure=None, background=None):
        self.geom=geom
        self.axes=axes
        self.exposure=exposure
        self.background=background
        
    def predict(self):
        spatial=np.repeat(self.geom.spatial[np.newaxis,:],len(self.axes.centers),axis=0)
        
        spectrum=self.axes.spectrum
        spectrum=np.repeat(spectrum[:,np.newaxis],spatial.shape[-2],axis=-1)
        spectrum=np.repeat(spectrum[:,:,np.newaxis],spatial.shape[-1],axis=-1)
        
        return spatial*spectrum*self.exposure+self.background
    
    def likelihood(self, counts, **kwargs):
        pred_cube=self.predict()
        pred_total = np.sum(pred_cube)
        # In order to avoid too large number for fatorial
        # use Stirling's Approximation here, which is 
        # ln(n!) = n * ln(n) - n (first order approximation)

        # Only elements greater than 10 is valid
        # TODO : use masked array to do condition slicing
        # counts_masked = np.ma.masked_array(counts, mask=mask_large)
        mask_large = counts > 10
        counts_masked = np.ma.masked_array(counts, mask=mask_large)


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
    
    def fit(self,counts,method='nelder-mead'):
        def fun(x):
            self.geom.sigma=x[0]
            self.axes.prefactor=x[1]
            self.axes.index=x[2]
            return -self.likelihood(counts)
        
        #return func(1e-10,-3.18,0.8)
        x0=[self.geom.sigma,self.axes.prefactor,self.axes.index]
        res = minimize(fun, x0, method=method, options={'xatol': 1e-8, 'disp': True})
        return res
    
    @property
    def flux(self):
        pred_spectrum=self.predict()
        return pred_spectrum.sum(axis=(1,2))
    
    def plot_flux(self,counts):
        plt.plot(self.axes.centers,self.flux,label='pred')
        plt.plot(self.axes.centers,counts.sum(axis=(1,2)),label='real')
        #plt.xticks(self.axes.centers,np.around(self.axes.centers,4))
        plt.xlabel('Energy')
        plt.ylabel('counts')
        plt.title('flux')