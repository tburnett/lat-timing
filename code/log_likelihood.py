"""Implement maximizaion of weighted log likeliood
"""

import os, sys
import numpy as np
from scipy import optimize
from numpy import linalg


class LogLike(object):
    """ implement Kerr Eqn 2 """
    
    def __init__(self, time, exposure_fraction, weights):

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
        