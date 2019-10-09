"""Manage photon data, S/C history, GTI, effective area
"""

import os, glob, pickle
import healpy
import numpy as np
import pandas as pd

from scipy.integrate import simps
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
from exposure import EffectiveArea
import keyword_options

mission_start = Time('2001-01-01T00:00:00', scale='utc')
day = 24*3600.

def MJD(met):
    "convert MET to MJD"
    return (met/day + mission_start.mjd)
def UT(mjd):
    " convert MJD value to ISO date string"
    t=Time(mjd, format='mjd')
    t.format='iso'; t.out_subfmt='date_hm'
    return t.value


class Data(object):
    """Manage the file-based data sets for photons, livetime, and space craft orientation history
    """
    
    defaults=(
        ('radius', 5, 'cone radius for selection [deg]'),
        ('verbose',2,'verbosity level'),
        'File locations, as glob patterns',
        ('data_file_pattern','$FERMI/data/P8_P305/time_info/month_*.pkl', 'monthly photon data files'),
        ('ft2_file_pattern', '/nfs/farm/g/glast/g/catalog/P8_P305/ft2_20*.fits', 'yearly S/C history files'),
        ('gti_file_pattern', '$FERMI/data/P8_P305/yearly/*.fits', 'glob pattern for yearly Files with GTI info'),
        ('cos_theta_max', 0.4, 'cosine of maximum S/C theta'),
        ('z_max', 100, 'maximum angle between zenith and S/C bore'),
        ('mjd_range', None, 'default MJD limits'),
        'For estimate of exposure',
        ('base_spectrum', 'lambda E: (E/1000)**-2.1', 'Spectrum to use'),
        ('energy_domain', np.logspace(2,5,13), 'energy bins for exposure'),
        ('test', True, 'test mode'),
        ('ignore_gti', False, ''),
    )

    @keyword_options.decorate(defaults)
    def __init__(self, setup, **kwargs):
        """
        """
        keyword_options.process(self,kwargs)
        self.__dict__.update(setup.__dict__) # copy stuff from client

        # generate lists of files to process
        data_files, gti_files, ft2_files = self._check_files(self.mjd_range)

        if self.test:
            self.gti = self._process_gti(gti_files)
            # convert each to a DataFrame
            self.exposure =  self._process_ft2(ft2_files,self.gti)
            self.photon_data = self._process_data(data_files, self.gti)

        # temporary for comparing with old code
        self.ft2_files = ft2_files
        self.gti_files = gti_files

            
    def _check_files(self, mjd_range):
        """ return lists of files to process
        """
    
        data_files = sorted(glob.glob(os.path.expandvars(self.data_file_pattern)))
        assert len(data_files)>0, 'No files found using pattern {}'.format(self.data_file_pattern)
        if self.verbose>2:
            gbtotal = np.array([os.stat(filename).st_size for filename in data_files]).sum()/2**30
            print(f'Found {len(data_files)} monthly photon data files, with {gbtotal:.1f} GB total')

        ft2_files = sorted(glob.glob(os.path.expandvars(self.ft2_file_pattern)))
        gti_files = sorted(glob.glob(os.path.expandvars(self.gti_file_pattern)))
        assert len(ft2_files)>0 and len(gti_files)>0, 'Fail to find FT2 or GTI files'
        assert len(ft2_files)--len(gti_files), 'expect to find same number of FT2 and GTI files'


        if mjd_range is not None:
            mjd_range = np.array(mjd_range).clip(mission_start.mjd, None)
            tlim =Time(mjd_range, format='mjd').datetime
            ylim,mlim = np.array([t.year for t in tlim])-2008, np.array([t.month for t in tlim])-1
            year_range = ylim.clip(0, len(gti_files)) + np.array([0,1])
            month_range = (ylim*12+mlim-8).clip(0,len(data_files)) + np.array([0,2]) #add 2 months?
            if self.verbose>0:
                print(f'From MJD range {mjd_range} select years {year_range}, months {month_range}')
        else:
            if self.verbose>0:
                print('Loading all found data')

        return (data_files if mjd_range is None else data_files[slice(*month_range)], 
                gti_files  if mjd_range is None else gti_files[slice(*year_range)], 
                ft2_files  if mjd_range is None else ft2_files[slice(*year_range)],
            )
    
    def _get_limits(self, a):
        return np.searchsorted(a, self.mjd_range)  

    def _process_data(self, data_files, gti): 

        dflist=[] 
        if self.verbose>0: print(f'Loading data from {len(data_files)} months ', end='')
        for filename in data_files:
            d = self._load_photon_data(filename, gti)
            if d is not None:
                dflist.append(d)
                if self.verbose>1: print('.', end='')
            else:
                if self.verbose>1: print('x')
                break
        assert len(dflist)>0, '\nNo photon data found?'
        df = pd.concat(dflist, ignore_index=True)
        if self.verbose>0:
            print(f'\n\tSelected {len(df)} photons within {self.radius}'\
                  f' deg of  ({self.l:.2f},{self.b:.2f})')
            ta,tb = df.iloc[0].time, df.iloc[-1].time
            print(f'\tDates: {UT(ta):16} - {UT(tb)}'\
                f'\n\tMJD  : {ta:<16.1f} - {tb:<16.1f}')  
        return df 

    def _load_photon_data(self, filename, gti, nside=1024):
        """Read in, process a file generated by binned_data.ConvertFT1.time_record
        
        return DataFrame with times, band id, distance from center
            
        parameters:
            filename : file name
            gti   : GTI object for filtering data
            nside : for healpy

        returns:
            DataFrame with columns:
                band : from input, energy and event type  
                time : Mission Elapsed Time in s. (double)
                delta : distance from input position (deg, float32)
        """       
        l,b,radius = self.l, self.b, self.radius      
        with open(filename,'rb') as f:
            d = pickle.load( f ,encoding='latin1')
            tstart = d['tstart']
            if MJD(tstart) > self.mjd_range[1]:
                return None
            df = pd.DataFrame(d['timerec'])
        # cartesian vector from l,b for healpy stuff    
        cart = lambda l,b: healpy.dir2vec(l,b, lonlat=True) 

        # use query_disc to get photons within given radius of position
        center = healpy.dir2vec(l,b, lonlat=True) #cart(l,b)
        ipix = healpy.query_disc(nside, cart(l,b), np.radians(radius), nest=False)
        incone = np.isin(df.hpindex, ipix)
        
        # times: convert to double, add to start, convert to MJD filtter with GTI
        t = MJD(np.array(df.time[incone],float)+tstart)
        in_gti = gti(t)
        if np.sum(in_gti)==0:
            return None

        # convert position info to just distance from center             
        ll,bb = healpy.pix2ang(nside, df.hpindex[incone][in_gti],  nest=False, lonlat=True)
        t2 = np.array(np.sqrt((1.-np.dot(center, cart(ll,bb)))*2), np.float32) 

        out_df = pd.DataFrame(np.rec.fromarrays(
            [df.band[incone][in_gti], t[in_gti], np.degrees(t2)], 
            names='band time delta'.split()))

        return out_df
    
    def write(self, filename):
        """ write to a file
        """
        out = dict(
            name=self.name, 
            galactic=(self.l,self.b), 
            radius=self.radius,
            time_data=self.data_df.to_records(index=False),
        )        
        pickle.dump(out, open(filename, 'wb'))

    def _exposure(self, livetime, pcosine):
        """return exposure calculated for each pair in livetime and cosines arrays
        uses effective area 
        """
        assert len(livetime)==len(pcosine), 'expect equal-length arrays'
        # get a set of energies and associated weights from a trial spectrum
        base_spectrum = eval(self.base_spectrum) #lambda E: (E/1000)**-2.1 
        assert base_spectrum(1000)==1.
        edom = self.energy_domain
        wts = base_spectrum(edom) 

        # effectivee area function from 
        ea = EffectiveArea()

        # a table of the weighted for each pair in livetime and pcosine arrays
        rvals = np.empty([len(wts),len(pcosine)]) 
        for i,(en,wt) in enumerate(zip(edom,wts)): 
            faeff,baeff = ea([en],pcosine) 
            rvals[i] = (faeff+baeff)*wt

        aeff = simps(rvals,edom,axis=0)/simps(wts,edom)
        return (aeff*livetime)

    def _process_ft2(self, ft2_files, gti):
        """Process set of FT2 files, with S/C history data
        Generate a data set with fields:
            start, stop : start  and stop times in MJD
            exposure    : calculated exposure using effective area
        Apply selection cuts for 
            - overall selected MJD range
            = GTI
            - limits on angles between selected direction and s/c z-axis and zenith
         """
        # combine the files into a DataFrame with following fields besides START and STOP (lower case for column)
        fields    = ['LIVETIME','RA_SCZ','DEC_SCZ', 'RA_ZENITH','DEC_ZENITH'] 
        if self.verbose>1:
            print(f'Processing {len(ft2_files)} S/C history (FT2) files')
        sc_data=[]
        for filename in ft2_files:
            with fits.open(filename) as hdu:
                scdata = hdu['SC_DATA'].data
                # get times to check against MJD limits and GTI
                start, stop = [MJD(np.array(scdata.START, float)), 
                               MJD(np.array(scdata.STOP, float))]
                if self.mjd_range is not None:
                    a,b=  self.mjd_range
                    if start[0]>b or stop[-1]<a:
                        print(f'Reject file {filename}: not in range' )
                        continue
                # apply GTI--want the interval fully inside
                in_gti = np.logical_and(gti(start) , gti(stop))
                if self.verbose>2:
                    print(f'\tfile {filename}: {len(start)} entries, {sum(in_gti)} in GTI')
                t = [('start', start[in_gti]), ('stop',stop[in_gti])]+\
                    [(field.lower(), np.array(scdata[field][in_gti],np.float32)) for field in fields ]                   
                sc_data.append( pd.DataFrame(dict(t) ) )
        df = pd.concat(sc_data, ignore_index=True)
    
        # calculate cosines with respect to sky direction
        sc = SkyCoord(self.l,self.b, unit='deg', frame='galactic').fk5
        ra_r,dec_r = np.radians(sc.ra.value), np.radians(sc.dec.value)
        sdec, cdec = np.sin(dec_r), np.cos(dec_r)

        def cosines( ra2, dec2):
            ra2_r =  np.radians(ra2.values)
            dec2_r = np.radians(dec2.values)
            return np.cos(dec2_r)*cdec*np.cos(ra_r-ra2_r) + np.sin(dec2_r)*sdec

        pcosines = cosines(df.ra_scz,    df.dec_scz)
        zcosines = cosines(df.ra_zenith, df.dec_zenith)

        # mask out entries too close to zenith, or too far away from ROI center
        mask =   (pcosines >= self.cos_theta_max) & (zcosines>=np.cos(np.radians(self.z_max)))
        if self.verbose>1:
            print(f'\tFound {len(mask):,} S/C entries:  {sum(mask):,} remain after zenith and theta cuts')
        dfm = df.loc[mask,:]
        livetime = dfm.livetime.values
        self.dfm = dfm ##############debug
        # apply MJD range if present. note times in MJD
        start, stop = dfm.start,dfm.stop
        lims = slice(None)
        if self.mjd_range is not None:
            a, b = self._get_limits(start)
            if a>0 or b<len(start):
                if self.verbose>1:
                    print(f'\tcut from {len(start):,} to {a} - {b}, or {b-a:,} entries after MJD range selection')
                dfm = dfm.iloc[a:b]
                lims = slice(a,b)

        expose = self._exposure(livetime[lims], pcosines[mask][lims])
        return pd.DataFrame(dict(start=start[lims],stop=stop[lims], exposure=expose))
   
    def _process_gti(self, gti_files):
        """Combine the GTI intervals that fall within the gti range 
        Return a function that tests a list of times 
        """
        if self.verbose>1:
            print(f'Processing {len(gti_files)} GTI files ... ', end='')
        starts=[] 
        stops=[]
        for i, ft1 in enumerate(gti_files):
            with fits.open(ft1) as hdu: 
                gti_data = hdu['GTI'].data
                start = gti_data.START
                if i>0:
                    assert start[0]>= stops[-1][-1], f'file {ft1} has start time not following preceding file'
                starts.append(start)
                stops.append( gti_data.STOP)
        start = MJD(np.concatenate(starts))
        stop  = MJD(np.concatenate(stops))

        if self.verbose>1:
            livetime = (stop-start).sum()
            print( f' {len(gti_files)} files, {len(start)} intervals with'\
                   f' {int(livetime):,} days live time')

        if self.mjd_range is not None:
            a, b = self._get_limits(start)
            if a>0 or b<len(start):
                if self.verbose>1:
                    print(f'\tcut from {len(start):,} to {a} - {b}, or {b-a:,} entries after MJD range selection')

                sel = slice(a,b)
            else: sel=slice(None)

        class GTI(object):
            """ functor class that tests for being in the GTI range
            """
            def __init__(self, start, stop, ignore=False):
                # prepare single merged array with even, odd entries start and stop
                a,b =start, stop
                self.ignore=ignore
                self.fraction = np.sum(b-a)/(b[-1]-a[0])
                assert len(a)==len(b)
                self.g = np.array([a,b]).T.flatten()
                assert np.sum(np.diff(self.g)<0)==0, 'Bad GTI ordering'

            def __call__(self, time):
                # use digitize to find if in good/bad interval by odd/even
                if self.ignore: return np.ones(len(time)).astype(bool)
                x = np.digitize(time, self.g)
                return np.bitwise_and(x,1).astype(bool)

            def __repr__(self):
                return f'{self.__class__.__name__} MjD range: {self.g[0]:.2f}-{self.g[-1]:.2f}, good fraction {self.fraction:.2f} '

        gti =  GTI(start[sel],stop[sel], self.ignore_gti)
        if self.verbose>1:
            print(f'\t{gti}')
        return gti
        