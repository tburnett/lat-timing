"""Implement maximizaion of weighted log likeliood
"""

import os, sys
import numpy as np
from scipy import (optimize, linalg)
from scipy.linalg import (LinAlgError, LinAlgWarning)

class LogLike(object):
    """ implement Kerr Eqn 2 for a single interval, or cell"""
    
    def __init__(self, cell):
        self.__dict__.update(cell)

        self.estimate= [0, 0]
        
    def __call__(self, pars ):
        """ evaluate the log likelihood (not really used)"""

        alpha, beta= pars if len(pars)>1 else (pars[0], 0.)
        loglike= np.sum( np.log(1 + alpha*self.w + beta*(1-self.w) )) - alpha*self.S - beta*self.B

        return loglike

    def __repr__(self):
        return f'''{self.__class__}
        time {self.t:.3f} exposure {self.exp:.2e} S {self.S:.2f}, B {self.B:.2f}
        {len(self.w)} weights, mean {self.w.mean():.2f}, std {self.w.std():.2f}'''
        
    def gradient(self, pars ):
        """gradient of the loglikelihood with respect to alpha and beta, or just alpha"""
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
        
    def rate(self, fix_beta=False, debug=False):
        """Return signal rate and its error"""
        
        try:
            s = self.solve(fix_beta)
            if s is None:
                return None
            h = self.hessian(s)
        
            v = 1./h[0] if fix_beta else linalg.inv(h)[0,0]
            return (1+s[0]), np.sqrt(v)
        except (LinAlgError, LinAlgWarning, RuntimeWarning) as msg:
            if debug:
                print(f'Fit error, cell {self},\n\t{msg}')
            return None
            
    def minimize(self,   fix_beta=False, **fmin_kw):
        """Minimize the -Log likelihood """
        kw = dict(disp=False)
        kw.update(**fmin_kw)
        f = lambda pars: -self(pars)
        return optimize.fmin_cg(f, self.estimate[0:1] if fix_beta else self.estimate, **kw)

    def solve(self, fix_beta=False, debug=False, **fit_kw):
        """Solve non-linear equation(s) from setting gradient to zero 
        note that the hessian is a jacobian
        """
        kw = dict(factor=2, xtol=1e-3, fprime=self.hessian)
        kw.update(**fit_kw)
        estimate = self.estimate[0:1] if fix_beta else self.estimate
        try:
            ret = optimize.fsolve(self.gradient, estimate , **kw)   
        except RuntimeWarning as msg:
            if debug:
                print(f'Runtime fsolve warning for cell {self}\n\t {msg}')
            return None
        return np.array(ret)
        