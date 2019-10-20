"""Imported from godot/core.py
* convert to python 3
"""

import numpy as np
import pylab as plt

from collections import deque

from scipy.stats import chi2
from scipy.integrate import simps,cumtrapz
from scipy.optimize import fmin,fsolve,fmin_tnc,brentq
from scipy.interpolate import interp1d


def met2mjd(times,mjdref=51910+7.428703703703703e-4):
    times = np.asarray(times,dtype=np.float128)
    return times*(1./86400)+mjdref

def mjd2met(times,mjdref=51910+7.428703703703703e-4):
    times = (np.asarray(times,dtype=np.float128)-mjdref)*86400
    return times


class Cell(object):
    """ Encapsulate the concept of a Cell, specifically a start/stop
    time, the exposure, and set of photons contained within."""

    def __init__(self,tstart,tstop,exposure,photon_times,photon_weights,
            source_to_background_ratio):
        self.tstart = tstart
        self.tstop = tstop
        self.exp = exposure
        self.ti = np.atleast_1d(photon_times)
        self.we = np.atleast_1d(photon_weights)
        self.SonB = source_to_background_ratio

    def sanity_check(self):
        if self.exp==0:
            assert(len(self.ti)==0)
        assert(len(self.ti)==len(self.we))
        assert(np.all(self.ti >= self.tstart))
        assert(np.all(self.ti < self.tstop))

    def get_tmid(self):
        return 0.5*(self.tstart+self.tstop)


class CellLogLikelihood(object):

    def __init__(self,cell):
        self.cell = cell
        self.ti = cell.ti
        self.we = cell.we
        self.iwe = 1-self.we
        self.S = cell.exp
        self.B = self.S/cell.SonB
        self._tmp1 = np.empty_like(self.we)
        self._tmp2 = np.empty_like(self.we)
        self._last_beta = 0.

    def log_likelihood(self,alpha):
        # NB the minimum defined alpha is between -1 and 0 according to
        # amin = (wmax-1)/wmax
        t1,t2 = self._tmp1,self._tmp2
        np.multiply(self.we,alpha,out=t1)
        np.add(t1,self.iwe,out=t1)
        np.log(t1,out=t2)
        return np.sum(t2)-(self.S*alpha)

    def log_profile_likelihood_approx(self,alpha):
        """ Profile over the background level under assumption that the
        background variations are small (few percent).
        
        For this formulation, we're using zero-based parameters, i.e
        flux = F0*(1+alpha)."""

        alpha = alpha-1
        t1,t2 = self._tmp1,self._tmp2
        np.multiply(self.we,alpha,out=t1)
        t1 += 1.
        # t1 is now 1+alpha*we
        np.divide(self.iwe,t1,out=t2)
        #t2 is now 1-w/1+alpha*w
        Q = np.sum(t2)
        t2 *= t2
        R = np.sum(t2)
        # for reference
        beta_hat = (Q-self.B)/R

        t1 += beta_hat*(1-self.we)
        np.log(t1,out=t2)
        #return np.sum(t2) + 0.5*(Q-B)**2/R -alpha*self.exp
        return np.sum(t2) - (1+alpha)*self.S - (1+beta_hat)*self.B

    def log_profile_likelihood(self,alpha,beta_guess=1):
        """ Profile over the background level with no restriction on
        amplitude.  Use a prescription similar to finding max on alpha.

        For this formulation, we're using zero-based parameters.
        """

        beta = self._last_beta = self.get_beta_max(
                alpha,guess=beta_guess)-1
        alpha = alpha-1
        t1,t2 = self._tmp1,self._tmp2
        np.multiply(self.we,alpha-beta,out=t1)
        np.add(1.+beta,t1,out=t1)
        # t1 is now 1+beta+we(alpha-beta)
        np.log(t1,out=t1)
        return np.sum(t1)-self.S*(1+alpha)-self.B*(1+beta)

    def get_likelihood_grid(self,amax=2,bmax=2,res=0.01):
        na = int(round(amax/res))+1
        nb = int(round(bmax/res))+1
        agrid = np.linspace(0,amax,na)-1
        bgrid = np.linspace(0,bmax,nb)-1
        rvals = np.empty((na,nb))
        S,B = self.S,self.B
        t1,t2 = self._tmp1,self._tmp2
        iw = self.iwe
        for ia,alpha in enumerate(agrid):
            t2[:] = alpha*self.we+1
            for ib,beta in enumerate(bgrid):
                np.multiply(beta,iw,out=t1)
                np.add(t2,t1,out=t1)
                np.log(t1,out=t1)
                rvals[ia,ib] = np.sum(t1) -B*(1+beta)
            rvals[ia,:] -= S*(1+alpha) 
        return agrid+1,bgrid+1,rvals

    def log_full_likelihood(self,p):
        """ Likelihood for both source and background normalizations.
        """

        alpha,beta = p
        alpha -= 1
        beta -= 1
        t1,t2 = self._tmp1,self._tmp2
        np.multiply(alpha-beta,self.we,out=t1)
        np.add(1+beta,t1,out=t1)
        np.log(t1,out=t1)
        return np.sum(t1) - self.S*(1+alpha) -self.B*(1+beta)

    def fmin_tnc_func(self,p):
        alpha,beta = p
        alpha -= 1
        beta -= 1
        S,B = self.S,self.B
        t1 = np.multiply(self.we,alpha-beta,out=self._tmp1)
        t1 += 1+beta
        # above equivalent to
        # t1[:] = 1+beta*iw+alpha*self.we
        t2 = np.log(t1,out=self._tmp2)
        logl = np.sum(t2) - S*alpha -B*beta
        np.divide(self.we,t1,out=t2)
        grad_alpha = np.sum(t2)-S
        np.divide(self.iwe,t1,out=t2)
        grad_beta = np.sum(t2)-B
        return -logl,[-grad_alpha,-grad_beta]

    def fmin_fsolve(self,p):
        alpha,beta = p
        alpha -= 1
        beta -= 1
        S,B = self.S,self.B
        w = self.we
        iw = 1-self.we
        t1,t2 = self._tmp1,self._tmp2
        t1[:] = 1+beta*iw+alpha*w
        grad_alpha = np.sum(w/t1)-S
        grad_beta = np.sum(iw/t1)-B
        print(p,grad_alpha,grad_beta)
        return [-grad_alpha,-grad_beta]

    def fmin_fsolve_jac(self,p):
        pass

    def f1(self,alpha):
        w = self.we
        return np.sum(w/(alpha*w+(1-w)))-self.S

    def f1_profile(self,alpha):
        w = self.we
        a = alpha-1
        S,B = self.S,self.B
        t1 = self._tmp1
        t2 = self._tmp2
        t1[:] = (1-w)/(1+a*w)
        np.multiply(t1,t1,out=t2)
        t3 = w/(1+a*w)
        T2 = np.sum(t2)
        beta_hat = (np.sum(t1)-B)/T2
        t1_prime = np.sum(t3*t1)
        t2_prime = np.sum(t3*t2)
        beta_hat_prime = (2*beta_hat*t2_prime-t1_prime)/T2
        return np.sum((w+beta_hat_prime*(1-w))/(1+a*w+beta_hat*(1-w))) -S -beta_hat_prime*B

    def f2(self,alpha):
        w = self.we
        t = alpha*w+(1-w)
        return -np.sum((w/t)**2)

    def f3(self,alpha):
        w = self.we
        t = alpha*w+(1-w)
        return np.sum((w/t)**3)

    def nr(self,guess=1,niter=6):
        """ Newton-Raphson solution to max."""
        # can precalculate alpha=0, alpha=1 for guess, but varies depending
        # on TS, might as well just stick to 1 and use full iteration
        a = guess
        w = self.we
        iw = 1-self.we
        S = self.S
        t = np.empty_like(self.we)
        for i in range(niter):
            t[:] = w/(a*w+iw)
            f1 = np.sum(t)-S
            t *= t
            f2 = -np.sum(t)
            a = max(0,a - f1/f2)
        return a

    def halley(self,guess=1,niter=5):
        """ Hally's method solution to max."""
        a = guess
        w = self.we
        iw = 1-self.we
        S = self.S
        t = np.empty_like(self.we)
        t2 = np.empty_like(self.we)
        for i in range(niter):
            t[:] = w/(a*w+iw)
            f1 = np.sum(t)-S
            np.multiply(t,t,out=t2)
            f2 = -np.sum(t2)
            np.multiply(t2,t,out=t)
            f3 = 2*np.sum(t)
            a = max(0,a - 2*f1*f2/(2*f2*f2-f1*f3))
        return a

    def get_max(self,guess=1,beta=1,profile_background=False,
            recursion_count=0):
        """ Find value of alpha that optimizes the log likelihood.

        Is now switched to the 0-based parameter convention.

        Use an empirically tuned series of root finding.
        """
        if profile_background:

            # I'm not sure if it's best to keep to a default guess or to
            # try it at the non-profile best-fit.
            #if guess == 1:
                #guess = self.get_max(profile_background=False)
            #if beta == 1:
                #beta = self.get_beta_max(guess)
            # NB -- removed the scale below as it seemed to work better
            # without it.  In other words, the scale is already ~1!
            # test TNC method
            rvals,nfeval,rc = fmin_tnc(self.fmin_tnc_func,[guess,beta],
                    bounds=[[0,None],[0,None]],disp=0,ftol=1e-3)
            # this is one possible way to check for inconsistency in max
            #if (guess == 1) and (rvals[0] > 10):
                #rvals,nfeval,rc = fmin_tnc(self.fmin_tnc_func,
                        #[rvals[0],beta],
                        #bounds=[[0,None],[0,None]],disp=0,ftol=1e-3)

            if not((rc < 0) or (rc > 2)):
                if (guess == 0) and (rvals[0] > 5e-2):
                    print('Warning, possible inconsistency.  Guess was 0, best fit value %.5g.'%(rvals[0]),'beta=',beta)
                return rvals

            # try a small grid to seed a search
            grid = np.asarray([0,0.1,0.3,0.5,1.0,2.0,5.0,10.0])
            cogrid = np.asarray([self.log_profile_likelihood(x) for x in grid])
            newguess = grid[np.argmax(cogrid)]
            newbeta = max(0.1,self.get_beta_max(newguess))

            rvals,nfeval,rc = fmin_tnc(self.fmin_tnc_func,
                    [newguess,newbeta],
                    bounds=[[0,None],[0,None]],disp=0,ftol=1e-3)
            if not((rc < 0) or (rc > 2)):
                return rvals
            else:
                print('Warning, trouble locating maximum with profile_background!  Results for this interval may be unreliable.')
            return rvals

        w = self.we
        iw = self.iwe
        S,B = self.S,self.B
        guess = guess-1
        beta = beta-1

        # check that the maximum isn't at flux=0 (alpha-1) with derivative
        a = -1
        t1,t2 = self._tmp1,self._tmp2
        t2[:] = 1+beta*iw
        t1[:] = w/(t2+a*w)
        if (np.sum(t1)-S) < 0:
            return 0
        else:
            a = guess

        # on first iteration, don't let it go to 0
        t1[:] = w/(t2+a*w)
        f1 = np.sum(t1)-S
        t1 *= t1
        f2 = np.sum(t1) # will include sign below
        a = a + f1/f2
        if a < 0-1:
            a = 0.2-1

        # second iteration more relaxed
        t1[:] = w/(t2+a*w)
        f1 = np.sum(t1)-S
        t1 *= t1
        f2 = np.sum(t1) # will include sign below
        a = a + f1/f2
        if a < 0.05-1:
            a = 0.05-1

        # last NR iteration allow 0
        t1[:] = w/(t2+a*w)
        f1 = np.sum(t1)-S
        t1 *= t1
        f2 = np.sum(t1) # will include sign below
        alast = a = max(0-1,a + f1/f2)

        # now do a last Hally iteration
        t1[:] = w/(t2+a*w)
        f1 = np.sum(t1)-S
        t1 *= t1
        f2 = np.sum(t1) # will include sign below
        t1 *= w/(t2+a*w)
        f3 = 2*np.sum(t1)
        a = max(0-1,a + 2*f1*f2/(2*f2*f2-f1*f3))
        
        # a quick check if we are converging slowly to try again or if
        # we started very from from the guess (large value)
        if (abs(a-alast)>0.05) or (abs(guess-a) > 10):
            if recursion_count > 2:
                return self.get_max_numerical()
                #raise ValueError('Did not converge!')
            return self.get_max(guess=a+1,beta=beta+1,
                    recursion_count=recursion_count+1)

        return a+1

    def get_max_numerical(self,guess=1,beta=1,profile_background=False,
            recursion_count=0):
        """ Find value of alpha that optimizes the log likelihood.

        Is now switched to the 0-based parameter convention.

        Use an empirically tuned series of root finding.
        """
        if profile_background:
            # TODO -- probably want to replace this with a better iterative
            # method, but for now, just use good ol' fsolve!

            # test TNC method
            rvals,nfeval,rc = fmin_tnc(self.fmin_tnc_func,[guess,beta],
                    bounds=[[0,None],[0,None]],disp=0,ftol=1e-3)
            if (rc < 0) or (rc > 2):
                print('Warning, best guess probably wrong.')
            return rvals

        w = self.we
        iw = self.iwe
        S,B = self.S,self.B
        guess = guess-1
        beta = beta-1

        # check that the maximum isn't at flux=0 (alpha-1) with derivative
        a = -1
        t1,t2 = self._tmp1,self._tmp2
        t2[:] = 1+beta*iw
        t1[:] = w/(t2+a*w)
        if (np.sum(t1)-S) < 0:
            return 0
        else:
            a = guess

        def f(a):
            t1[:] = w/(t2+a*w)
            return np.sum(t1)-S

        a0 = -1
        amax = guess
        for i in range(12):
            if f(amax) < 0:
                break
            a0 = amax
            amax = 2*amax+1
        return brentq(f,a0,amax,xtol=1e-3)+1

    def get_beta_max(self,alpha,guess=1,recursion_count=0):
        """ Find value of beta that optimizes the likelihood, given alpha.
        """
        if np.isnan(alpha):
            return 1
        #display = False
        alpha = alpha-1
        guess -= 1
        S,B = self.S,self.B
        w = self.we
        iw = self.iwe
        t,t2 = self._tmp1,self._tmp2

        # check that the maximum isn't at 0 (-1) with derivative
        t2[:] = 1+alpha*w
        t[:] = iw/(t2-iw)
        if (np.sum(t)-B) < 0:
            return 0
        else:
            b = guess
        #if display:
        #    print '0:',b+1

        # on first iteration, don't let it go to 0
        t[:] = iw/(t2+b*iw)
        f1 = np.sum(t)-B
        t *= t
        f2 = np.sum(t) # will include sign below
        b = b + f1/f2
        #if display:
        #    print '1:',b+1
        b = max(0.2-1,b)
        #if b < 0.2-1:
            #b = 0.2-1
        #if display:
        #    print '1p:',b+1

        # second iteration more relaxed
        t[:] = iw/(t2+b*iw)
        f1 = np.sum(t)-B
        t *= t
        f2 = np.sum(t) # will include sign below
        b = b + f1/f2
        #if display:
        #    print '2:',b+1
        b = max(0.05-1,b)
        #if b < 0.05-1:
            #b = 0.05-1
        #if display:
        #    print '2p:',b+1

        # last NR iteration allow 0
        # second iteration more relaxed
        t[:] = iw/(t2+b*iw)
        f1 = np.sum(t)-B
        t *= t
        f2 = np.sum(t) # will include sign below
        b = b + f1/f2
        #if display:
        #    print '3:',b+1
        #if b < 0-1:
            #b = 0.02-1
        b = max(0.02-1,b)
        #if display:
        #    print '3p:',b+1

        # replace last NR iteration with a Halley iteration to handle
        # values close to 0 better; however, it can result in really
        # huge values, so add a limiting step to it
        t[:] = iw/(t2+b*iw)
        f1 = np.sum(t)-B
        t *= t
        f2 = np.sum(t) # will include sign below
        t *= iw/(t2+b*iw)
        f3 = 2*np.sum(t)
        newb = max(0-1,b + 2*f1*f2/(2*f2*f2-f1*f3))
        #if display:
        #    print '4:',newb+1,b+1
        if abs(newb-b) > 10:
            blast = b = 2*b+1
        else:
            blast = b = newb
        #if display:
        #    print '4p:',b+1

        # now do a last Hally iteration
        t[:] = iw/(t2+b*iw)
        f1 = np.sum(t)-B
        t *= t
        f2 = np.sum(t) # will include sign below
        t *= iw/(t2+b*iw)
        f3 = 2*np.sum(t)
        b = max(0-1,b + 2*f1*f2/(2*f2*f2-f1*f3))
        #if display:
        #    print '5:',b+1

        # a quick check if we are converging slowly to try again or if
        # the final value is very large
        if (abs(b-blast)>0.05) or (abs(guess-b) > 10) or (b==-1):
            if recursion_count > 2:
                raise ValueError('Did not converge for alpha=%.5f!'%(
                    alpha+1))
            return self.get_beta_max(alpha+1,guess=b+1,
                    recursion_count=recursion_count+1)

        return b+1

    def get_beta_max_numerical(self,alpha,guess=1):
        alpha = alpha-1
        S,B = self.S,self.B
        w = self.we
        iw = self.iwe
        t,t2 = self._tmp1,self._tmp2

        # check that the maximum isn't at 0 (-1) with derivative
        t2[:] = 1+alpha*w
        t[:] = iw/(t2-iw)
        if (np.sum(t)-B) < 0:
            return 0

        def f(b):
            t[:] = iw/(t2+b*iw)
            return np.sum(t)-B

        b0 = -1
        bmax = guess-1
        for i in range(8):
            if f(bmax) < 0:
                break
            b0 = bmax
            bmax = 2*bmax+1
        return brentq(f,b0,bmax,xtol=1e-3)+1

    def get_logpdf(self,aopt=None,dlogl=20,npt=100,include_zero=False,
            profile_background=False):
        """ Evaluate the pdf over an adaptive range that includes the
            majority of the support.  Try to keep it to about 100 iters.
        """
        if profile_background:
            return self._get_logpdf_profile(aopt=aopt,dlogl=dlogl,npt=npt,
                    include_zero=include_zero)
        if aopt is None:
            aopt = self.get_max()
        we = self.we
        iw = self.iwe
        S,B = self.S,self.B
        t = self._tmp1
        amin = 0
        if aopt == 0:
            # find where logl has dropped, upper side
            llmax = np.log(iw).sum()
            # do a few NR iterations
            amax = max(0,-(llmax+dlogl)/(np.sum(we/(1-we))-S))
            for i in range(10):
                t[:] = amax*we+iw
                f0 = np.log(t).sum()-amax*S+dlogl-llmax
                f1 = np.sum(we/t)-S
                amax = amax - f0/f1
                if abs(f0) < 0.1:
                    break
        else:
            # find where logl has dropped, upper side
            t[:] = aopt*we + iw
            llmax = np.sum(np.log(t))-S*aopt
            # use Taylor approximation to get initial guess
            f2 = np.abs(np.sum((we/t)**2))
            amax = aopt + np.sqrt(2*dlogl/f2)
            # do a few NR iterations
            for i in range(5):
                t[:] = amax*we+iw
                f0 = np.log(t).sum()-amax*S+dlogl-llmax
                f1 = np.sum(we/t)-S
                amax = amax - f0/f1
                if abs(f0) < 0.1:
                    break
        if not include_zero:
            # ditto, lower side
            t[:] = aopt*we + iw
            # use Taylor approximation to get initial guess
            f2 = np.abs(np.sum((we/t)**2))
            amin = aopt - np.sqrt(2*dlogl/f2)
            # do a few NR iterations
            for i in range(5):
                t[:] = amin*we+iw
                f0 = np.log(t).sum()-amin*S+dlogl-llmax
                f1 = np.sum(we/t)-S
                amin = amin - f0/f1
                if abs(f0) < 0.1:
                    break
        amin = max(0,amin)

        dom = np.linspace(amin,amax,npt)
        cod = np.empty_like(dom)
        for ia,a in enumerate(dom):
            cod[ia] = self.log_likelihood(a) 

        # do a sanity check here
        acodmax = np.argmax(cod)
        codmax = cod[acodmax]
        if abs(codmax - llmax) > 0.05:
            aopt = dom[acodmax]
            return self.get_logpdf(aopt=aopt,dlogl=dlogl,npt=npt,
                    include_zero=include_zero)

        cod -= llmax
        return dom,cod

    def _get_logpdf_profile(self,aopt=None,dlogl=20,npt=100,
            include_zero=False):
        """ Evaluate the pdf over an adaptive range that includes the
            majority of the support.  Try to keep it to about 100 iters.
        """
        if aopt is None:
            aopt,bopt = self.get_max(profile_background=True)
        else:
            bopt = self.get_beta_max(aopt)

        # the algebra gets pretty insane here, so I think it's easier just
        # to find the range numerically

        amin = 0
        llmax = self.log_full_likelihood([aopt,bopt])

        f = lambda a:self.log_profile_likelihood(a)-llmax+dlogl
        a0 = aopt
        amin = 0
        amax = max(5,a0)
        # make sure upper range contains root
        for i in range(4):
            if f(amax) > 0:
                a0 = amax
                amax *= amax
        amax = brentq(f,a0,amax)
        if aopt > 0:
            if f(0) > 0:
                amin = 0
            else:
                amin = brentq(f,0,aopt)

        dom = np.linspace(amin,amax,npt)
        cod = np.empty_like(dom)
        self._last_beta = 0
        for ia,a in enumerate(dom):
            cod[ia] = self.log_profile_likelihood(a)
                    #beta_guess=self._last_beta+1)

        cod -= llmax
        return dom,cod

    def get_pdf(self,aopt=None,dlogl=20,npt=100,profile_background=False):
        dom,cod = self.get_logpdf(aopt=aopt,dlogl=dlogl,npt=npt,
                profile_background=profile_background)
        np.exp(cod,out=cod)
        return dom,cod*(1./simps(cod,x=dom))

    def get_ts(self,aopt=None,profile_background=False):
        if self.S == 0:
            return 0
        if aopt is None:
            aopt = self.get_max(profile_background=profile_background)
            print(aopt)
            if profile_background:
                aopt = aopt[0] # discard beta
            print(aopt)
        if aopt == 0:
            return 0
        func = self.log_profile_likelihood if profile_background else self.log_likelihood
        return 2*(func(aopt)-func(0))

    def get_flux(self,conf=[0.05,0.95],profile_background=False):
        aopt = self.get_max(profile_background=profile_background)
        if profile_background:
            aopt = aopt[0]
        dom,cod = self.get_pdf(aopt=aopt,
                profile_background=profile_background)
        amax = np.argmax(cod)
        func = self.log_profile_likelihood if profile_background else self.log_likelihood
        if abs(dom[amax]-aopt) > 0.1: # somewhat ad hoc
            # re-optimize
            aopt = self.get_max(guess=dom[amax],
                    profile_background=profile_background)
            if profile_background:
                aopt = aopt[0]
            if abs(dom[amax]-aopt) > 0.1: # somewhat ad hoc
                print('failed to obtain agreement, using internal version')
                aopt = dom[amax]
        ts = self.get_ts(aopt=aopt,profile_background=profile_background)
        cdf = cumtrapz(cod,dom,initial=0)
        cdf *= 1./cdf[-1]
        indices = np.searchsorted(cdf,conf)
        # do a little linear interpolation step here
        ihi,ilo = indices,indices-1
        m = (cdf[ihi]-cdf[ilo])/(dom[ihi]-dom[ilo])
        xconf = dom[ilo] + (np.asarray(conf)-cdf[ilo])/m
        return aopt,ts,xconf

    def get_profile_flux(self):
        """ Make an expansion of the likelihood assuming a small deviation
            of the source and background density from the mean, and
            return the flux estimator after profiling over the background.

            This has not been carefully checked, but does look sane from
            some quick tests.
        """
        N = len(self.we)
        # estimate of source background from exposure
        S,B = self.S,self.B
        W = np.sum(self.we)
        W2 = np.sum(self.we**2)
        t1 = W2-W
        t2 = t1 + (N-W)
        ahat = ((W-S)*t2 + (N-B-W)*t1) / (W2*t2 - t1**2)
        return ahat


class CellsLogLikelihood(object):
    """ A second attempt that attempts to do a better job of sampling the
        log likelihood.

        Talking through it -- if we don't sample the log likelihoods on
        a uniform grid, then for practicality if we use a value of alpha
        that is outside of the domain of one of the sub likelihoods, it
        is basically "minus infinity".  There are some contrived cases
        where you'd include something really significant with a different
        flux with a bunch of other significant things at a different flux.
        Then the formal solution would be in the middle, but it would be
        a terrible fit, so in terms of Bayesian blocks or anything else,
        you'd never want it.  So maybe it doesn't matter what its precise
        value is, and we can just return "minus infinity".

        This bears some checking, but for now, that's how it's coded up!

        What this means practically is that as we go along evaluating a
        fitness function, the bounds are always defined as the supremumm
        of the lower edges of the domains and the vice versa for the upper
        edges.  Still then have the problem of finding the maximum for
        irregular sampling.
    """
    def __init__(self,cells,profile_background=False):

        # construct a sampled log likelihoods for each cell
        self.clls = list(map(CellLogLikelihood,cells))
        npt = 200
        self._cod = np.empty([len(cells),npt])
        self._dom = np.empty([len(cells),npt])
        self.cells = cells

        for icll,cll in enumerate(self.clls):
            self._dom[icll],self._cod[icll] = cll.get_logpdf(
                    npt=npt,dlogl=30,profile_background=profile_background)
        self.profile_background = profile_background

        self.fitness_vals = None

    def sanity_check(self):
        dx = self._dom[:,-1]-self._dom[:,0]
        bad_x = np.ravel(np.argwhere(np.abs(dx) < 1e-3))
        print('Indices with suspiciously narrow support ranges: ',bad_x)
        ymax = self._cod.max(axis=1)
        bad_y = np.ravel(np.argwhere(np.abs(ymax)>0.2))
        print('Indices where there is a substantial disagreement in the optimized value of the log likelihood and the codomain: ', bad_y)
    
    def fitness(self,i0,i1):
        """ Return the maximum likelihood estimator and value for the
        cell slice i0:i1, define such that ith element is:

        0 -- cells i0 to i1
        1 -- cells i1+1 to i1
        ...
        N -- i1

        """

        # set bounds on domain
        npt = 1000
        a0,a1 = self._dom[i1-1][0],self._dom[i1-1][-1]
        dom = np.linspace(a0,a1,npt)

        # initialize summed log likelihoods
        rvals = np.empty((2,i1-i0))
        cod = np.zeros(npt)

        #for i in xrange(0,len(rvals)):
        for i in range(0,rvals.shape[1]):
            cod += np.interp(dom,self._dom[i1-1-i],self._cod[i1-1-i],
                    left=-np.inf,right=-np.inf)
            amax = np.argmax(cod)
            rvals[:,-(i+1)] = cod[amax],dom[amax]
            #rvals[-(i+1)] = cod[amax]
            #rvals[-(i+1)] = np.max(cod)

        return rvals*2

    def do_bb(self,prior=2):

        if self.fitness_vals is not None:
            return self._do_bb_cache(prior=prior)

        fitness = self.fitness

        ncell = len(self.cells)
        best = np.asarray([-np.inf]*(ncell+1))#np.zeros(nph+1)
        last = np.empty(ncell,dtype=int)
        fitness_vals = deque()
        last[0] = 0; best[0] = 0
        tmp = fitness(0,1)
        best[1] = tmp[0]-prior
        fitness_vals.append(tmp)
        for i in range(1,ncell):
            # form A(r) (Scargle VI)
            tmp = fitness(0,i+1)
            fitness_vals.append(tmp)
            a = tmp[0] - prior + best[:i+1]
            # identify last changepoint in new optimal partition
            rstar = np.argmax(a)
            best[i+1] = a[rstar]
            last[i] = rstar
        best_fitness = best[-1]

        cps = deque()
        last_index = last[-1]
        while last_index > 0:
            cps.append(last_index)
            last_index = last[last_index-1]
        indices = np.append(0,np.asarray(cps,dtype=int)[::-1])

        self.fitness_vals = fitness_vals

        # calculate overall variability TS
        var_dof = len(indices)-1
        var_ts = (best_fitness-tmp[0][0])+prior*var_dof

        return indices,best_fitness+len(indices)*prior,var_ts,var_dof,fitness_vals

    def _do_bb_cache(self,prior=2):

        fv = [x[0] for x in self.fitness_vals]

        ncell = len(self.cells)
        best = np.asarray([-np.inf]*(ncell+1))#np.zeros(nph+1)
        last = np.empty(ncell,dtype=int)
        last[0] = 0; best[0] = 0
        best[1] = fv[0]-prior
        for i in range(1,ncell):
            # form A(r) (Scargle VI)
            a = fv[i] - prior + best[:i+1]
            # identify last changepoint in new optimal partition
            rstar = np.argmax(a)
            best[i+1] = a[rstar]
            last[i] = rstar
        best_fitness = best[-1]

        cps = deque()
        last_index = last[-1]
        while last_index > 0:
            cps.append(last_index)
            last_index = last[last_index-1]
        indices = np.append(0,np.asarray(cps,dtype=int)[::-1])

        # calculate overall variability TS
        var_dof = len(indices)-1
        var_ts = (best_fitness-fv[-1][0])+prior*var_dof

        return indices,best_fitness+len(indices)*prior,var_ts,var_dof,self.fitness_vals

    def do_top_hat(self):
        """ Get a top-hat filtered representation of cells, essentially
            using the BB fitness function, but returning both the TS and
            the optimum value.
        """
        fitness_vals = self.do_bb()[-1]
        n = len(fitness_vals)
        rvals = np.empty((2,n,n))
        ix = np.arange(n)
        for iv,v in enumerate(fitness_vals):
            rvals[:,ix[:iv+1],ix[:iv+1]+n-(iv+1)] = v
            rvals[:,ix[:iv+1]+n-(iv+1),ix[:iv+1]] = np.nan
        # add an adjustment for dof to the TS map, and correct the flux
        rvals[0] += (np.arange(n)*2)[None,::-1]
        rvals[1] *= 0.5
        return rvals

    def get_flux(self,idx,conf=[0.05,0.95]):

        aguess = self._dom[idx][np.argmax(self._cod[idx])]
        if self.profile_background:
            bguess = self.clls[idx].get_beta_max(aguess)
        else:
            bguess = 1
        aopt = self.clls[idx].get_max(guess=aguess,beta=bguess,
                profile_background=self.profile_background)
        ## sanity check that aopt is close to the guess.  If not, it is
        ## wrong or else we extracted the domain/codomain incorrectly.
        #if abs(aopt[0]-aguess) > 0.1:
            #print aopt,aguess
            #print 'Warning! refinement of aguess failed.  Using internal version.'

        if self.profile_background:
            aopt = aopt[0]
        ts = self.clls[idx].get_ts(aopt=aopt,
                profile_background=self.profile_background)

        dom,cod = self._dom[idx],self._cod[idx].copy()
        np.exp(cod,out=cod)
        cod *= (1./simps(cod,x=dom))
        cdf = cumtrapz(cod,dom,initial=0)
        cdf *= 1./cdf[-1]
        indices = np.searchsorted(cdf,conf)
        # do a little linear interpolation step here
        ihi,ilo = indices,indices-1
        m = (cdf[ihi]-cdf[ilo])/(dom[ihi]-dom[ilo])
        xconf = dom[ilo] + (np.asarray(conf)-cdf[ilo])/m
        return aopt,ts,xconf

    def log_likelihood(self,alpha,slow_but_sure=False):
        if len(alpha) != len(self.clls):
            raise ValueError('Must provide a value of alpha for all cells!')
        if not hasattr(self,'_interpolators'):
            # populate interpolators on first call
            assert False, 'interpolator needed'
            #self._interpolators = [interp1d(d,c) for d,c in zip(self._dom,self._cod)]
        if not slow_but_sure:
            rvals = 0
            try:
                for a,i in zip(alpha,self._interpolators):
                    rvals += i(a)
                return rvals
            except ValueError:
                pass
        if self.profile_background:
            return sum((cll.log_profile_likelihood(a) for cll,a in zip(self.clls,alpha)))
        else:
            return sum((cll.log_likelihood(a) for cll,a in zip(self.clls,alpha)))

    def get_lightcurve(self,tsmin=4,plot_years=False,plot_phase=False,
            get_ts=False):
        """ Return a flux density light curve for the raw cells.
        """

        plot_phase = plot_phase ## THB not checked yet or isinstance(self,PhaseCellsLogLikelihood)

        # time, terr, yval, yerrlo,yerrhi; yerrhi=-1 if upper limit
        rvals = np.empty([len(self.clls),5])
        all_ts = np.empty(len(self.clls))
        for icll,cll in enumerate(self.clls):
            if cll.S==0:
                rvals[icll] = np.nan
            tmid = cll.cell.get_tmid()
            if plot_phase:
                terr = (tmid-cll.cell.tstart)
            else:
                terr = (tmid-cll.cell.tstart)/86400
                tmid = met2mjd(tmid)
                if plot_years:
                    tmid = (tmid-54832)/365 + 2009 
                    terr *= 1./365
            aopt,ts,xconf = self.get_flux(icll,conf=[0.16,0.84])
            ul = ts <= tsmin
            if ul:
                rvals[icll] = tmid,terr,xconf[1],0,-1
            else:
                rvals[icll] = tmid,terr,aopt,aopt-xconf[0],xconf[1]-aopt
            all_ts[icll] = ts

        if get_ts:
            return rvals,all_ts
        return rvals

    def plot_cells_bb(self,tsmin=4,fignum=2,clear=True,color='C3',
            plot_raw_cells=True,bb_prior=4,plot_years=False,
            no_bb=False,log_scale=False,
            plot_phase=False,ax=None):

        # NB might want to use a CellsLogLikelihood to avoid overhead of 3x
        # size on BB computation
        plot_phase = plot_phase or isinstance(self,PhaseCellsLogLikelihood)

        if ax is None:
            pl.figure(fignum)
            if clear:
                pl.clf()
            ax = pl.gca()

        if log_scale:
            ax.set_yscale('log')
        if plot_raw_cells:
            # time, terr, yval, yerrlo,yerrhi; yerrhi=-1 if upper limit
            rvals = np.empty([len(self.clls),5])
            for icll,cll in enumerate(self.clls):
                if cll.S==0:
                    rvals[icll] = np.nan
                tmid = cll.cell.get_tmid()
                if plot_phase:
                    terr = (tmid-cll.cell.tstart)
                else:
                    terr = (tmid-cll.cell.tstart)/86400
                    tmid = met2mjd(tmid)
                    if plot_years:
                        tmid = (tmid-54832)/365 + 2009 
                        terr *= 1./365
                aopt,ts,xconf = self.get_flux(icll,conf=[0.16,0.84])
                if ts <= tsmin:
                    rvals[icll] = tmid,terr,xconf[1],0,-1
                else:
                    rvals[icll] = tmid,terr,aopt,aopt-xconf[0],xconf[1]-aopt
            ul_mask = (rvals[:,-1] == -1) & (~np.isnan(rvals[:,-1]))
            t = rvals[ul_mask].transpose()
            ax.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C0',alpha=0.2,ls=' ',ms=3)
            t = rvals[~ul_mask].transpose()
            ax.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='o',color='C0',alpha=0.2,ls=' ',ms=3)
        else:
            rvals = None

        # now, do same for Bayesian blocks
        if not no_bb:
            bb_idx,bb_ts,var_ts,var_dof,fitness = self.do_bb(prior=bb_prior)
            #print(var_ts,var_dof)
            print(f'Variability significance: {chi2.sf(var_ts,var_dof):.2e}')
            bb_idx = np.append(bb_idx,len(self.cells))
            rvals_bb = np.empty([len(bb_idx)-1,5])
            for ibb,(start,stop) in enumerate(zip(bb_idx[:-1],bb_idx[1:])):
                cells = cell_from_cells(self.cells[start:stop])
                cll = CellLogLikelihood(cells)
                if cll.S==0:
                    rvals_bb[ibb] = np.nan
                    continue
                tmid = cll.cell.get_tmid()
                if plot_phase:
                    terr = (tmid-cll.cell.tstart)
                else:
                    terr = (tmid-cll.cell.tstart)/86400
                    tmid = met2mjd(tmid)
                    if plot_years:
                        tmid = (tmid-54832)/365 + 2009 
                        terr *= 1./365
                aopt,ts,xconf = cll.get_flux(conf=[0.16,0.84],
                        profile_background=self.profile_background)
                if ts <= tsmin:
                    rvals_bb[ibb] = tmid,terr,xconf[1],0,-1
                else:
                    rvals_bb[ibb] = tmid,terr,aopt,aopt-xconf[0],xconf[1]-aopt

            ul_mask = (rvals_bb[:,-1] == -1) & (~np.isnan(rvals_bb[:,-1]))
            t = rvals_bb[ul_mask].transpose()
            ax.errorbar(t[0],t[2],xerr=t[1],yerr=0.1*t[2],uplims=True,marker=None,color='C3',alpha=0.8,ls=' ',ms=3)
            t = rvals_bb[~ul_mask].transpose()
            ax.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='o',color='C3',alpha=0.8,ls=' ',ms=3)
        else:
            rvals_bb=None

        if plot_phase:
            ax.set_xlabel('Pulse Phase')
            ax.axis([0,1,pl.axis()[2],pl.axis()[3]])
        elif plot_years:
            ax.set_xlabel('Year')
        else:
            ax.set_xlabel('MJD')
        ax.set_ylabel('Relative Flux Density')
        return rvals,rvals_bb

    def get_bb_lightcurve(self,tsmin=4,plot_years=False,plot_phase=False,
            bb_prior=8):
        """ Return a flux density light curve for the raw cells.
        """

        plot_phase = False #### plot_phase or isinstance(self,PhaseCellsLogLikelihood)

        # now, do same for Bayesian blocks
        bb_idx,bb_ts,var_ts,var_dof,fitness = self.do_bb(prior=bb_prior)
        print(var_ts,var_dof)
        print('Variability significance: ',chi2.sf(var_ts,var_dof))
        bb_idx = np.append(bb_idx,len(self.cells))
        rvals_bb = np.empty([len(bb_idx)-1,5])
        for ibb,(start,stop) in enumerate(zip(bb_idx[:-1],bb_idx[1:])):
            cells = cell_from_cells(self.cells[start:stop])
            cll = CellLogLikelihood(cells)
            if cll.S==0:
                rvals_bb[ibb] = np.nan
                continue
            tmid = cll.cell.get_tmid()
            if plot_phase:
                terr = (tmid-cll.cell.tstart)
            else:
                terr = (tmid-cll.cell.tstart)/86400
                tmid = met2mjd(tmid)
                if plot_years:
                    tmid = (tmid-54832)/365 + 2009 
                    terr *= 1./365
            aopt,ts,xconf = cll.get_flux(conf=[0.16,0.84],
                    profile_background=self.profile_background)
            if ts <= tsmin:
                rvals_bb[ibb] = tmid,terr,xconf[1],0,-1
            else:
                rvals_bb[ibb] = tmid,terr,aopt,aopt-xconf[0],xconf[1]-aopt

        return rvals_bb




        super(PhaseCellsLogLikelihood,self).__init__(cells)

        # extend the domain/comain by one wrap on each side
        self._dom = np.concatenate([self._dom]*3)
        self._cod = np.concatenate([self._cod]*3)
        self._orig_cells = cells
        self.cells = [c.copy(phase_offset=-1) for c in cells]
        self.cells += cells
        self.cells += [c.copy(phase_offset=+1) for c in cells]
        self.profile_background = False
                

def cell_from_cells(cells):
    """ Return a single Cell object for multiple cells."""

    cells = sorted(cells,key=lambda cell:cell.tstart)
    tstart = cells[0].tstart
    we = np.concatenate([c.we for c in cells])
    ti = np.concatenate([c.we for c in cells])
    exp = np.sum((c.exp for c in cells))
    ### THB slight mod to deal with apparent round-off for me
    #if not np.all(np.asarray([c.SonB for c in cells])==cells[0].SonB):
    if np.array([x.SonB for x in cells]).std()>1e-6:
        raise Exception('Cells do not all have same source flux!')
    return Cell(cells[0].tstart,cells[-1].tstop,exp,ti,we,cells[0].SonB)

def plot_clls_lc(rvals,ax=None,scale='linear',min_mjd=None,max_mjd=None,
        ul_color='C1',meas_color='C0'):
    """ Make a plot of the output lc CellsLogLikelihood.get_lightcurve.
    """
    if min_mjd is not None:
        mask = rvals[:,0] >= min_mjd
        rvals = rvals[mask,:]
    if max_mjd is not None:
        mask = rvals[:,0] <= max_mjd
        rvals = rvals[mask,:]
    if ax is None:
        ax = plt.gca()
    ax.set_yscale(scale)
    ul_mask = (rvals[:,-1] == -1) & (~np.isnan(rvals[:,-1]))
    t = rvals[ul_mask].transpose()
    ax.errorbar(t[0],t[2],xerr=t[1],yerr=0.2*(1 if scale=='linear' else t[2]),uplims=True,marker=None,color=ul_color,alpha=0.5,ls=' ',ms=3)
    t = rvals[~ul_mask].transpose()
    ax.errorbar(t[0],t[2],xerr=t[1],yerr=[t[3],t[4]],marker='o',color=meas_color,alpha=0.5,ls=' ',ms=3)
    ax.set_xlabel('MJD')
    ax.set_ylabel('Relative Flux')
