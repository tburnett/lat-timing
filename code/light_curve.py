"""Manage the binned time data to produce light curves, etc.

"""
import numpy as np
import pylab as plt
import pandas as pd
import os, sys

from scipy import (optimize, linalg)
from scipy.linalg import (LinAlgError, LinAlgWarning)
import keyword_options

data=None # data_managment.Data obect for access to name, verbose, etc.

class LightCurve(object):
    """ In the language of Kerr, manage a set of cells
    """
    defaults=(
        ('min_exp', 0.3, 'mimimum exposure factor'),
    )
    @keyword_options.decorate(defaults)
    def __init__(self, binned_weights, **kwargs):
        """Load binned data
        parameters:
            binned_weights : an iterable object that is a list of dicts; expect each to have
                 keys t, exp, w, S, B
            min_exp : filter by exposure factor
        """
        keyword_options.process(self,kwargs)
        global data
        data=binned_weights.data
        self.cells = [LogLike(ml) for ml in binned_weights if ml['exp']>self.min_exp] 
        if data.verbose>0:
            print(f'Loaded {len(self)} / {len(binned_weights)} cells with exposure > {self.min_exp} for light curve analysis')
    
         
    def __repr__(self):
        return f'{self.__class__} {len(self.cells)} cells loaded'
    
    def __getitem__(self, i):
        return self.cells[i]
    
    def __len__(self):
        return len(self.cells)
    
    def fit(self, fix_beta=False, no_ts=True):
        """Perform fits to all intervals, set a DataFrame with results
        """
        r=[]; bad=[]; good=[]; ts=[]
        for ll in self:
            result = ll.rate(fix_beta=fix_beta, no_ts=no_ts)
            if result is None: 
                bad.append(ll)
            else:
                s, sig, t = result
                r.append([s, sig])
                ts.append(t)
                good.append(ll)
        fits = np.array(r)
        self.bad =bad # save list for later study
        if data.verbose>0:
            print(f'Fits: {len(good)} good, {len(bad)} failed ')
        self.fit_df=  pd.DataFrame([[c.t for c in good],
                              [c.exp for c in good],fits[:,0], fits[:,1],ts],
                      index='time exp rate sigma ts'.split()).T
        
    def rate_plot(self,fix_beta=False, title=None, ax=None): 
        if not hasattr(self, 'fit_df'):
            self.fit(fix_beta)

        df=self.fit_df
        t = df.time
        y=  df.rate.values.clip(0,4)
        dy= df.sigma.values.clip(0,4)
        
        if not ax:
            fig, ax = plt.subplots(figsize=(12,4))
        else:
            fig =ax.figure
        ax.errorbar(x=t, y=y, yerr=dy, fmt='+')
        ax.axhline(1., color='grey')
        ax.set(xlabel='MJD', ylabel='relative rate')
        ax.set_title(title or data.source_name)
        ax.grid(alpha=0.5)
        
    def fit_hists(self, title=None, **hist_kw):
        """ Generate set of histograms of rate, error, pull, and maybe TS
        """
        hkw = dict(log=True, histtype='stepfilled',lw=2, edgecolor='blue', facecolor='lightblue')
        hkw.update(hist_kw)

        df = self.fit_df
        fig, (ax1,ax2,ax3)= plt.subplots(1,3, figsize=(12,2.5))
        x = df.time
        y = df.rate
        yerr = df.sigma

        def shist(ax, x,  xlim, nbins, label, xlog=False): 
            def space(xlim, nbins=50):
                if xlog:
                    return np.logspace(np.log10(xlim[0]), np.log10(xlim[1]))
                return np.linspace(xlim[0], xlim[1], nbins+1)
            info = f'mean {x.mean():6.3f}\nstd  {x.std():6.3f}'
            ax.hist(x.clip(*xlim), bins=space(xlim, nbins), **hkw)
            ax.set(xlabel=label, xscale='log' if xlog else 'linear', ylim=(0.8,None))
            ax.text(0.65, 0.84, info, transform=ax.transAxes,fontdict=dict(size=10, family='monospace')) 
            ax.grid(alpha=0.5)
            return ax

        shist(ax1, y, (0.2, 5), 25, 'relative rate', xlog=True).axvline(1, color='grey')
        shist(ax2, yerr, (1e-2, 0.3), 25, 'sigma', xlog=True)
        shist(ax3, (y-1)/yerr, (-6,6), 25,'pull').axvline(0,color='grey')
        fig.suptitle(title or data.source_name+' fit summary')
    
"""Implement maximizaion of weighted log likeliood
"""
class LogLike(object):
    """ implement Kerr Eqn 2 for a single interval, or cell"""
    
    def __init__(self, cell):
        
        self.__dict__.update(cell)
        self.estimate= [0, 0]
        
    def __call__(self, pars ):
        """ evaluate the log likelihood """

        alpha, beta= pars if len(pars)>1 else (pars[0], 0.)
        loglike= np.sum( np.log(1 + alpha*self.w + beta*(1-self.w) )) - alpha*self.S - beta*self.B

        return loglike

    def __repr__(self):
        return f'''{self.__class__}
        time {self.t:.3f} exposure {self.exp:.2f} S {self.S:.0f}, B {self.B:.0f}
        {len(self.w)} weights, mean {self.w.mean():.2f}, std {self.w.std():.2f}'''
        
    def gradient(self, pars ):
        """gradient of the log likelihood with respect to alpha and beta, or just alpha"""
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
        """return Hessian matrix (1 or 2 D according to pars) from explicit second derivatives
        Note this is also the Jacobian of the gradient.
        """
        w = self.w
        fixed_beta = len(pars)==1
        if fixed_beta:
            alpha = pars[0]
            D = 1 + alpha*w 
            return [np.sum((w/D)**2)]
        else:
            alpha, beta= pars
            Dsq = (1 + alpha*w + beta*(1-w))**2
            a, b, c = np.sum(w**2/Dsq), np.sum(w*(1-w)/Dsq), np.sum((1-w)**2/Dsq)
            return np.array([[a,b], [b,c]])
        
    def rate(self, fix_beta=False, debug=False, no_ts=True):
        """Return signal rate and its error"""
        #TODO: return upper limit if TS<?
        try:
            s = self.solve(fix_beta)
            if s is None:
                return None
            h = self.hessian(s)
        
            v = 1./h[0] if fix_beta else linalg.inv(h)[0,0]
            ts = None if no_ts else (0 if s[0]<=-1 else 2*(self(s)-self([-1,s[1]])))
            return (1+s[0]), np.sqrt(v), ts
        
        except (LinAlgError, LinAlgWarning, RuntimeWarning) as msg:
            if debug or data.verbose>2:
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
            if debug or data.verbose>2:
                print(f'Runtime fsolve warning for cell {self}, \n\t {msg}')
            return None
        return np.array(ret)
        
    def loglikeplot(ll,fix_beta=False, xlim=(-0.2,0.2),ax=None, title=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4,2))
        else: fig=ax.figure
        dom = np.linspace(*xlim)
        v, s, ts = ll.rate(fix_beta=fix_beta, debug=True)
        if fix_beta:
            f = lambda x: ll([x])
            a, beta = v-1, 0.
        else:
            a, beta = ll.solve(fix_beta, debug=True)
            assert abs(a-v+1)<1e-6, f'{a},{v}, {a-v+1}'
            f = lambda x: ll([x, beta])
        ax.plot(dom, list(map(f,dom)) )

        ax.plot(a, f(a), 'or')
        ax.plot([a-s, a+s], [f(a-s), f(a+s)], '-k',lw=2)
        for x in (a-s,a+s):
            ax.plot([x,x], [f(x)-0.1, f(x)+0.1], '-k',lw=2)
        ax.plot(a, f(a)-0.5, '-ok', ms=10)
        ax.grid()
        ax.set(title=title, xlim=xlim, ylim=(f(a)-4, f(a)+0.2), 
               ylabel='log likelihood', xlabel=r'$\alpha$')
        