"""process weights for timed data
"""

import os, sys
import numpy as np
from scipy import optimize
from numpy import linalg
import pandas as pd

import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

import corner

#local
from data_management import Data
import keyword_options

class Main(object):

    defaults=(
        ('verbose', 1,),
        ('source_name', 'Geminga','name of the source'),
        ('mjd_range', (51910, 54800),'MJD limits to use'),
        ('weight_file', '../data/geminga_weights.pkl', 'location of pointlike weight file'),
    )
    
    @keyword_options.decorate(defaults)
    def __init__(self,  **kwargs):
        keyword_options.process(self, kwargs)

        sc = SkyCoord.from_name(self.source_name).galactic
        self.__dict__.update(l=sc.l.value, b=sc.b.value) 

        # load the data and produce the photon data dataframe
        self.data = data= Data(self)

        # add the weights to the photon dataframe
        wtd = data.add_weights(self.weight_file)
        # get info from weights file
        vals = [wtd[k] for k in 'model_name roi_name source_name source_lb '.split()]
        lbformat = lambda lb: '{:.2f}, {:.2f}'.format(*lb)
        self.model_info='\n  {}\n  {}\n  {}\n  {}'.format(vals[0], vals[1], vals[2], lbformat(vals[3]))

        # add a energy band column, filter out photons with NaN         
        gd = data.photon_data
        gd.loc[:,'eband']=(gd.band.values//2).clip(0,7)

        ok = np.logical_not(pd.isna(gd.weight))
        self.photons = gd.loc[ok,:]
        
    def weights(self):
        w = self.data.photon_data.weight
        good = np.logical_not(np.isnan(w))
        return w[good]

    
    def solve(self,  exposure_fraction, estimate=[0,0], fix_beta=False, **fmin_kw):
        """SOlve for alpha and beta for all of current selected data set
           TODO: extend to set of data segments
        
        parameters:
           exposure_fraction : Contribution to total exposure
           estimate : initial values for (alpha,beta)
           fix_beta : if True, only fit for alpha
           
        returns:
             fits for alpha,beta
        """
        return LogLike(self.weights(), exposure_fraction).solve(estimate, fix_beta, **fmin_kw)

        
    def corner_plot(self, weighted=False, **kwargs):
        """produce a "corner" plot.
        """
        fig, axx=plt.subplots(3,3, figsize=(10,10))
        corner.corner(self.photons['eband radius weight'.split()], 
                    range=[(-0.75,7.25),(0,5),(0,1)],
                    bins=(16,20,20),
                    label_kwargs=dict(fontsize=14), 
                    labels=['Energy band index', 'Radius [deg]', 'weight'],
                    weights=self.photons.weight if weighted else None,
                    show_titles=False, top_ticks=True, color='blue', fig=fig, 
                    hist_kwargs=dict(histtype='stepfilled', facecolor='lightgrey',edgecolor='blue', lw=2),
                      **kwargs);
        fig.set_facecolor('white')
        use_weights='weighted' if weighted else 'not weighted'
        fig.suptitle(f'{self.source_name} analysis\n  {len(self.photons)} events\n  {use_weights} \nModel info'+self.model_info,
                    fontsize=14, ha='left',
                    fontdict=dict(family='monospace'), #didn't work
                    )
        

class LogLike(object):
    """ implemtn Kerr Eqn 2 """
    
    def __init__(self, weights,  exposure_fraction):

        self.f = f = exposure_fraction
        self.w = w = weights 
        self.S = f * np.sum(w)
        self.B = f * np.sum(1-w)
        # alpha,beta starting point for fitting
        self.estimate= [1/f-1, 1/f-1]
        
    def __call__(self, pars ):

        alpha, beta= pars if len(pars)>1 else (pars[0], 0)
        loglike= np.sum( np.log(1 + alpha*self.w + beta*(1-self.w)  ))\
                    - alpha*self.S - beta*self.B
        return -loglike

    def gradient(self, pars ):
        w = self.w
        fixed_beta = len(pars)==1
        if fixed_beta:
            
            alpha =  pars[0] 
            D = 1 + alpha*w
            return np.sum(w/D) - self.S
        else:
            alpha, beta = pars
            D =  1 + alpha*w + beta*(1-w)
            da = np.sum(w/D) - self.S
            db = np.sum((1-w)/D) - self.B
            return [da,db]  
        
    def hessian(self, pars):
        """reuturn Hessian matrix from explicit second derivatives"""
        w = self.w
        fixed_beta = len(pars)==1
        if fixed_beta:
            alpha = pars[0]
            D = 1 + alpha*w 
            return [np.sum(w/D)]
        else:
            alpha, beta= pars
            Dsq = (1 - alpha*w + beta*(1-w))**2
            a, b, c = np.sum(w**2/Dsq), np.sum(w*(1-w)/Dsq), np.sum((1-w)**2/Dsq)
            return np.array([[a,b], [b,c]])
        
    def rate(self, fix_beta=False):
        """Return Normalized rate and variance matrix"""
        s = self.solve(fix_beta)
        self.fit_pars = s # for reference
        h = self.hessian(s)
        v = 1./h[0] if fix_beta else linalg.inv(h) 
        return (1+s)*self.f, v*self.f
            
    def minimize(self,   fix_beta=False, **fmin_kw):
        """Minimize the -Log likelihood """
        kw = dict(disp=False)
        kw.update(**fmin_kw)
        return optimize.fmin_cg(self, self.estimate[0:1] if fix_beta else self.estimate, **kw)

    def solve(self, fix_beta=False, **fit_kw):
        """Solve non-linear equation(s) from  setting gradient to zero """
        kw = dict()
        kw.update(**fit_kw)
        ret = optimize.newton_krylov(self.gradient,
                                      self.estimate[0:1] if fix_beta else self.estimate, **kw)   
        return np.array(ret)
        

        