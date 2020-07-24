"""Manage photon data, S/C history, GTI, effective area
Also add weights
"""

import os, glob, pickle
import healpy
import numpy as np
import pandas as pd

from scipy.integrate import simps
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord

from . effective_area import EffectiveArea
from . binner import BinnedWeights

from utilities import keyword_options

#mission_start = Time('2001-01-01T00:00:00', scale='utc').mjd
# From a FT2 file header
# MJDREFI =               51910. / Integer part of MJD corresponding to SC clock S
# MJDREFF =  0.00074287037037037 / Fractional part of MJD corresponding to SC cloc
mission_start = 51910.00074287037
day = 24*3600.
first_data=54683

def MJD(met):
    "convert MET to MJD"
    return (mission_start + met/day  )
def UTC(mjd):
    " convert MJD value to ISO date string"
    t=Time(mjd, format='mjd')
    t.format='iso'; t.out_subfmt='date_hm'
    return t.value


class TimedData(object):
    """Manage the file-based data sets for photons, livetime, and space craft orientation history
    """
    
    defaults=(
        ('source_name', 'unnamed', 'name for source'),

        ('verbose',3,'verbosity level'),
        
        'File locations, as glob patterns',
        ('data_file_pattern', '$HOME/work/lat-data/binned',
                                    #'$FERMI/data/P8_P305/time_info/month_*.pkl', 
                                    'monthly photon data files'),
        ('ft2_file_pattern',  '$HOME/work/lat-data/ft2/*.fits',  
                                  #'/nfs/farm/g/glast/g/catalog/P8_P305/ft2_20*.fits', 
                                    'yearly S/C history files'),
        ('gti_file_pattern',   '$HOME/work/lat-data/binned/*.fits',
                                #'$FERMI/data/P8_P305/yearly/*.fits', 
                                    'glob pattern for yearly Files with GTI info'),
        ('effective_area_path',      '$HOME/work/lat-data/aeff', 
                                    'where to find AEFF IRF'),

        ('energy_edges', np.logspace(2,6,17), 'expected eneegy bins'),
        
        'S/C limits',
        ('cos_theta_max', 0.4, 'cosine of maximum S/C theta'),
        ('z_max', 100, 'maximum angle between zenith and S/C bore'),
        
        'Data selection, binning',
        ('mjd_range', None, 'default MJD limits'),
        ('radius', 5, 'cone radius for selection [deg]'),
        ('energy_range', (100.,1e6), 'Selected energy range in MeV'),
        ('interval', 10, 'default binning step'),
        
        'For estimate of exposure',
        ('base_spectrum', 'lambda E: (E/1000)**-2.1', 'Spectrum to use'),
        #('energy_domain', 'np.logspace(2,5,13)', 'energy bins for exposure calculation (expresson)'),
        ('bins_per_decade',4, 'binning for exposure intefral'),
        ('nside', 1024, 'HEALPix nside which was used to bin photon data positions'),
        ('nest',   False, 'HEALPix used RING'),
        ('ignore_gti', False, ''),
    )

    @keyword_options.decorate(defaults)
    def __init__(self, setup, **kwargs):
        """
        setup : a dict with at least source_name, l,b 
        """
        keyword_options.process(self,kwargs)
        self.__dict__.update(setup.__dict__) # copy stuff from client

        # generate lists of files to process
        data_files, gti_files, ft2_files = self._check_files(self.mjd_range)

        self.gti = self._process_gti(gti_files)
        # convert each to a DataFrame
        self.exposure =  self._process_ft2(ft2_files,self.gti)
        photon_data = self._process_data(data_files, self.gti)

        if photon_data is None:
            print( 'No photon data??')
            return
        # set up DataFrame with all photons, and default binning
        self.photon_data =self._check_photons(self.exposure, photon_data)      
        self.edges = self._default_bins()
        self.binned_exposure = self.get_binned_exposure(self.edges)
           
    def __repr__(self):
        b = self.edges
        years = (b[-1]-b[0])/365.25
        return  f'Source "{self.source_name}" at (l,b)= ({self.l:.2f},{self.b:.2f}), radius= {self.radius}  '\
                f'\n     Data class: {self.__class__.__name__}  '\
                f'\n     {len(b)} cells, size {self.interval} day(s),'\
                f' from {b[0]:.1f} to {b[-1]:.1f} ({years:.1f} years).  '
       
    def _default_bins(self):
        #set up default bins from exposure; adjust stop to come out even
        # round to whole day

        start = np.round(self.exposure.start.values[0])
        stop =  np.round(self.exposure.stop.values[-1])
        if self.mjd_range is None:
            self.mjd_range = (start,stop)
        if self.interval >0:
            step = self.interval
            nbins = int(round((stop-start)/step))
            time_bins = np.linspace(start, stop, nbins+1)
            if self.verbose>0:
                print(f'Default binning: {nbins} intervals of {step} days, '\
                      f'in range ({time_bins[0]:.1f}, {time_bins[-1]:.1f})')
        else:
            a, b = self.get_contiguous_exposures()
            edges = np.empty(len(a)+1)
            edges[0] = a[0]
            edges[1:-1] = 0.5*(a[1:]+b[:-1])
            edges[-1]= b[-1]
            time_bins = edges
        return time_bins 

    def _process_weights(self):
        # add the weights to the photon dataframe
        wtd = self.add_weights(self.weight_file)
        # get info from weights file
        vals = [wtd[k] for k in 'model_name roi_name source_name source_lb '.split()]
        lbformat = lambda lb: '{:.2f}, {:.2f}'.format(*lb)
        self.model_info='\n  {}\n  {}\n  {}\n  {}'.format(vals[0], vals[1], vals[2], lbformat(vals[3]))

        # add a energy band column, filter out photons with NaN         
        gd = self.photon_data
        gd.loc[:,'eband']=(gd.band.values//2).clip(0,7)

        ok = np.logical_not(pd.isna(gd.weight))
        self.photons = gd.loc[ok,:]
        
    def binned_weights(self, bins=None, contiguous=False):
        """ 
        Parameter:
            bins : None | float | array
                if None, use defaults
                Otherwise an array of MJD bin edges
            contiguous : bool
                if True ignore bins arg and use a list of contiguous exposure intervals
                
        Returns: a BinnedWeight object for access to each set of binned weights
            The object can be indexed, or used in a for loop
            bw[i] returns a  dict (t, tw, fexp, w, S, B)
            where t   : bin center time (MJD)
                  tw  : bin width in days (assume 1 if not preseent)
                  fexp: associated fractional exposure
                  w   : array of weights for the time range
                  S,B : predicted source, background counts for this bin
            """
        if contiguous:
            assert bins is None, 'contiguous selected'
            a, b = self.get_contiguous_exposures()
            edges = np.empty(len(a)+1)
            edges[0] = a[0]
            edges[1:-1] = 0.5*(a[1:]+b[:-1])
            edges[-1]= b[-1]
            bins = edges
        return BinnedWeights(self, bins)     

    def get_binned_exposure(self, time_bins):

        # get stuff from photon data, exposure calculation
        exp   = self.exposure.exposure.values
        estart= self.exposure.start.values
        estop = self.exposure.stop.values
        
        #use cumulative exposure to integrate over larger periods
        cumexp = np.concatenate(([0],np.cumsum(exp)) )

        # get index into tstop array of the bin edges
        edge_index = np.searchsorted(estop, time_bins)
        # return the exposure integrated over the intervals
        return np.diff(cumexp[edge_index])
        
    def add_weights(self, filename):
        """Add weights to the photon data
        """
        
        # load a pickle containing weights, generated by pointlike
        assert os.path.exists(filename),f'File {filename} not found.'
        with open(filename, 'rb') as file:
            wtd = pickle.load(file, encoding='latin1')
        assert type(wtd)==dict, 'Expect a dictionary'
        test_elements = 'energy_bins pixels weights nside model_name radius order roi_name'.split()
        assert np.all([x in wtd.keys() for x in test_elements]),f'Dict missing one of the keys {test_elements}'

        pos = wtd['source_lb']
        if self.verbose>0:
            print(f'Adding weights from file {os.path.realpath(filename)}')
            print(f'Found weights for {wtd["source_name"]} at ({pos[0]:.2f}, {pos[1]:.2f})')
        # extract pixel ids and nside used
        wt_pix   = wtd['pixels']
        nside_wt = wtd['nside']
    
        # merge the weights into a table, with default nans
        # indexing is band id rows by weight pixel columns
        # append one empty column for photons not in a weight pixel
        # calculated weights are in a dict with band id keys        
        wts = np.full((32, len(wt_pix)+1), np.nan, dtype=np.float32)    
        weight_dict = wtd['weights']
        for k in weight_dict.keys():
            wts[k,:-1] = weight_dict[k]   

        # get the photon pixel ids, convert to NEST and right shift them 
        photons = self.photon_data
        photon_pix = healpy.ring2nest(self.nside, photons.pixel.values)
        to_shift = 2*int(np.log2(self.nside/nside_wt)); 
        shifted_pix =   np.right_shift(photon_pix, to_shift)
        bad = np.logical_not(np.isin(shifted_pix, wt_pix)) 
        if self.verbose>0:
            print(f'\t{sum(bad)} / {len(bad)} photon pixels are outside weight region')
        if sum(bad)==len(bad):
            raise Exception('No weights found')
        shifted_pix[bad] = 12*nside_wt**2 # set index to be beyond pixel indices

        # find indices with search and add a "weights" column
        # (expect that wt_pix are NEST ordering and sorted) 
        weight_index = np.searchsorted(wt_pix,shifted_pix)
        band_index = photons.band.values
        # final grand lookup -- isn't numpy wonderful!
        photons.loc[:,'weight'] = wts[tuple([band_index, weight_index])] 
        if self.verbose>0:
            print(f'\t{sum(np.isnan(photons.weight.values))} weights set to NaN')
        return wtd # for reference   
    
    def get_contiguous_exposures(self, max_interval=10 ):
        """Combine exposure intervals
        """
        
        t0s = self.exposure.start.values
        t1s = self.exposure.stop.values
        break_mask = (t0s[1:]-t1s[:-1])>max_interval/day 
        break_starts = t1s[:-1][break_mask]
        break_stops = t0s[1:][break_mask]
        # now assumble the complement
        good_starts = np.empty(len(break_starts)+1)
        good_stops = np.empty_like(good_starts)
        good_starts[0] = t0s[0]
        good_starts[1:] = break_stops
        good_stops[-1] = t1s[-1]
        good_stops[:-1] = break_starts
        if self.verbose>1:
            print(f'Generate list of contiguous exposures:\n'\
                  f'  WIth max interval {max_interval} s, combine {len(t0s):,} exposure entries to {len(good_stops):,} ')
        return good_starts, good_stops
        
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
            mjd_range = np.array(mjd_range).clip(first_data, None)
            tlim =Time(mjd_range, format='mjd').datetime
            ylim,mlim = np.array([t.year for t in tlim])-2008, np.array([t.month for t in tlim])-1
            year_range = ylim.clip(0, len(gti_files)) + np.array([0,1])
            month_range = (ylim*12+mlim-8).clip(0,len(data_files)) + np.array([0,2]) #add 2 months?
            if self.verbose>1:
                print(f'From MJD range {mjd_range} select years {year_range}, months {month_range}')
        else:
            if self.verbose>2:
                print('Loading all found data')

        self.mjd_range = mjd_range # save for reference

        return (data_files if mjd_range is None else data_files[slice(*month_range)], 
                gti_files  if mjd_range is None else gti_files[slice(*year_range)], 
                ft2_files  if mjd_range is None else ft2_files[slice(*year_range)],
            )
    
    def _get_limits(self, a):
        return np.searchsorted(a, self.mjd_range)  

    def _process_data(self, data_files, gti): 

        edom = eval(self.energy_domain)
        dflist=[] 
        if self.verbose>0: print(f'Loading data from {len(data_files)} months ', end='')
        for filename in data_files:
            d = self._load_photon_data(filename, gti)
            if d is not None:
                dflist.append(d)
                if self.verbose>1: print('.', end='')
            else:
                if self.verbose>1: print('x', end='')
                continue

        assert len(dflist)>0, '\nNo photon data found?'
        df = pd.concat(dflist, ignore_index=True)
        if self.verbose>0:
            print(f'\n\tSelected {len(df):,} photons within {self.radius}'\
                  f' deg of  ({self.l:.2f},{self.b:.2f})'\
                  f' with energies {edom[0]/1e3}-{edom[-1]/1e3:.0e} Gev')
            ta,tb = df.iloc[0].time, df.iloc[-1].time
            print(f'\tDates: {UTC(ta):16} - {UTC(tb)}'\
                f'\n\tMJD  : {ta:<16.1f} - {tb:<16.1f}')  
        return df 

    def _load_photon_data(self, filename, gti):
        """Read in, process a file generated by binned_data.ConvertFT1.time_record
        
        return DataFrame with times, band id, distance from center
            
        parameters:
            filename : file name
            gti   : GTI object for filtering data

        returns:
            DataFrame with columns:
                band : from input, energy and event type  
                time : Mission Elapsed Time in MJD (double)
                radius : radial distance from input position (deg, float32)
        """       
        l,b,radius = self.l, self.b, self.radius    
        if self.verbose>2:
            print(f'loading file {filename}')  
        with open(filename,'rb') as f:
            d = pickle.load( f ,encoding='latin1')
            tstart = d['tstart']
            if self.mjd_range is not None and MJD(tstart) > self.mjd_range[1]:
                return None
            df = pd.DataFrame(d['timerec'])

        # cartesian vector from l,b for healpy stuff    
        cart = lambda l,b: healpy.dir2vec(l,b, lonlat=True) 

        # use query_disc to get photons within given radius of position
        center = healpy.dir2vec(l,b, lonlat=True) #cart(l,b)
        ipix = healpy.query_disc(self.nside, cart(l,b), np.radians(radius), nest=False)
        incone = np.isin(df.hpindex, ipix)
        
        # times: convert to double, add to start, convert to MJD 
        t = MJD(np.array(df.time[incone],float)+tstart)
        in_gti = gti(t)
        if np.sum(in_gti)==0:
            return None

        # generate  radial distance from center from position 
        hpindex =  df.hpindex[incone][in_gti]           
        ll,bb = healpy.pix2ang(self.nside, hpindex,  nest=False, lonlat=True)
        t2 = np.array(np.sqrt((1.-np.dot(center, cart(ll,bb)))*2), np.float32) 

        out_df = pd.DataFrame(np.rec.fromarrays(
            [df.band[incone][in_gti], t[in_gti], hpindex, np.degrees(t2)], 
            names='band time pixel radius'.split()))

        return out_df
    
    def _check_photons(self, exposure, photon_data):
        # find range covered by exposure
        etime = photon_data.time.values #  note, in MJD units
        start, stop = exposure.start.values, exposure.stop.values
        imin,imax = np.searchsorted(etime, [start[0], stop[-1]])
        setime = etime[imin:imax]
        df = photon_data.iloc[imin:imax]
        if self.verbose>1:
            print(f'Check to see if photons have valid exposure: find {len(setime):,} photons within live time range')

        # get associated live time index for each photon
        lt_index = np.searchsorted(stop, setime) # before which stop
        # make sure past corresponding start
        tdiff = setime - start[lt_index]
        in_bin = tdiff>=0
        if self.verbose>1:
            print(f'\texclude {sum(~in_bin):,} not in an exposure bin--{sum(in_bin):,} remain.')
        etime_ok = setime[in_bin]
        
        # make debug df
        #self.debug_df=df.loc[~in_bin:'time']
        return df.loc[in_bin,:] 

    def write(self, filename):
        """ write to a file
        """
        dd = dict(
            name=self.name, 
            source_name=self.name,
            galactic=(self.l,self.b), 
            radius=self.radius,
            photon_data=self.photon_data.to_records(index=False),
            exposure=self.exposure.to_records(index=False),
            edges=self.edges,
            binned_exposure=self.binned_exposure,
            interval = self.interval,
            )   
              
        with open(filename, 'wb') as out:
            pickle.dump(dd, out)   

    def _exposure(self, livetime, pcosine):
        """return exposure calculated for each pair in livetime and cosines arrays
        uses effective area 
        """
        assert len(livetime)==len(pcosine), 'expect equal-length arrays'
        
        # get a set of energies and associated weights from a trial spectrum
 
        emin,emax = self.energy_range
        loge1=np.log10(emin); loge2=np.log10(emax)
        
        edom=np.logspace(loge1, loge2, int((loge2-loge1)*self.bins_per_decade+1))
        if self.verbose>2:
            print(f'exposure using energy domain {edom}')
        base_spectrum = eval(self.base_spectrum) #lambda E: (E/1000)**-2.1 
        assert base_spectrum(1000)==1.
        wts = base_spectrum(edom) 

        # effectivee area function from 
        ea = EffectiveArea(file_path= self.effective_area_path)

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
                # apply GTI to bin center (avoid edge effects?)
                in_gti = gti(0.5*(start+stop))
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

        sel = slice(None)
        if self.mjd_range is not None:
            a, b = self._get_limits(start)
            if a>0 or b<len(start):
                if self.verbose>1:
                    print(f'\tcut from {len(start):,} to {a} - {b}, or {b-a:,} entries after MJD range selection')
                sel = slice(a,b)


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
                return  f'{self.__class__.__name__} MjD range: {self.g[0]:.2f}-{self.g[-1]:.2f}'\
                        f', good fraction {self.fraction:.2f} '

        gti =  GTI(start[sel],stop[sel], self.ignore_gti)
        if self.verbose>1:
            print(f'\t{gti}')
        return gti

    def __getitem__(self, i):
        """ get info for ith time bin and return number
       
        """
#         k = self.edges       
#         wts = self.weights[k[i]:k[i+1]]
#         exp=self.fexposure[i]

#         return dict(
#                 t=self.bin_centers[i], # time
#                 exp=exp*self.N,        # exposure as a fraction of mean, for filtering
#                 w=wts,
#                 S= exp*self.S,
#                 B= exp*self.B,               
#                 )

    # def __len__(self):
    #     return self.N

    def test_plots(self):
        """Make a set of plots of exposure, counts, properties of weights
        """
        import matplotlib.pyplot as plt
        """  plots of properties of the weight distribution"""
        fig, axx = plt.subplots(5,1, figsize=(12,8), sharex=True,
                                         gridspec_kw=dict(hspace=0,top=0.95),)
        times=[]; vals = []
        for cell in self:
            t, e, w = [cell[q] for q in 't exp w'.split()]
            if e==0:
                continue
            times.append(t)
            vals.append( (e, len(w), len(w)/e , w.mean(), np.sum(w**2)/sum(w)))
        vals = np.array(vals).T
        for ax, v, ylabel in zip(axx, vals,
                            ['rel exp','counts','count rate', 'mean weight', 'rms/mean weight']):
            ax.plot(times, v, '+b')
            ax.set(ylabel=ylabel)
            ax.grid(alpha=0.5)
        axx[-1].set(xlabel='MJD')
        fig.suptitle(self.source_name)

class TimedDataX(TimedData):
    """ Read back file generated by TimedData.write
    """

    def __init__(self, filename):
        with open(filename, 'rb') as inp:
            pkl = pickle.load(inp)
        keys = list(pkl.keys())
        # convert recarray objects back to DataFrame
        for key in keys:
            if isinstance(pkl[key], np.recarray):
                pkl[key] = pd.DataFrame(pkl[key])
        self.__dict__.update(pkl)
        self.source_name = self.name
        self.l,self.b = self.galactic
        if not hasattr(self, 'interval'): self.interval = 1 # should have saved


class TimedDataArrow(TimedData):
    """Subclass of TimedData that uses Arrow parquet storage, basically HDF5
       Also 
    
    """
    
    defaults = TimedData.defaults\
            +(('parquet_root', '/nfs/farm/g/glast/u/burnett/analysis/lat_timing/data/',''),
              ('photon_dataset' ,'photon_dataset','parquet dataset'),
              ('tstart_file', 'tstart.pkl', 'dict of tstart values'),
              ('nest', True, 'HEALPix NEST if true, not RING indexing'),
            )

    
    @keyword_options.decorate( defaults)                    
    def __init__(self, setup, **kwargs):
        keyword_options.process(self, kwargs)
        infile = self.parquet_root+'/'+self.tstart_file
        with open(infile, 'rb') as inp:
            tstart_dict = pickle.load(inp)
        if self.verbose>0:
            print(f'Read {infile} with tstart values')  

        self.photon_data_source = dict(tstart_dict=tstart_dict, 
                                       dataset=self.parquet_root+'/'+self.photon_dataset)
        super().__init__(setup, **kwargs)
        
    def _check_files(self, mjd_range):
        assert mjd_range is None, 'MJD range not supported'
        
        ft2_files = sorted(glob.glob(os.path.expandvars(self.ft2_file_pattern)))
        gti_files = sorted(glob.glob(os.path.expandvars(self.gti_file_pattern)))
        assert len(ft2_files)>0 and len(gti_files)>0, 'Failed to find FT2 or GTI files'
        return self.photon_data_source, gti_files, ft2_files
    

    def _process_data(self, dummy1, dummy2): 
        import pyarrow.parquet as pq

        # cone geometry stuff: get corresponding pixels and center vector
        l,b,radius = self.l, self.b, self.radius  
        cart = lambda l,b: healpy.dir2vec(l,b, lonlat=True) 
        conepix = healpy.query_disc(self.nside, cart(l,b), np.radians(radius), nest=self.nest)
        center = healpy.dir2vec(l,b, lonlat=True)
        
        ebins = self.energy_edges
        ecenters = np.sqrt(ebins[:-1]*ebins[1:]); 
        band_limits = 2*np.searchsorted(ecenters, self.energy_range) if self.energy_range is not None else None
        
        def load_photon_data(table, tstart):
            """For a given month table, select photons in cone, add tstart to times, 
            return DataFrame with band, time, pixel, radius
            """
            allpix = np.array(table.column('nest_index'))
   

            def cone_select(allpix, conepix, shift=None):
                """Fast cone selection using NEST and shift
                """
                if shift is None:
                    return np.isin(allpix, conepix)
                assert self.nest, 'Expect pixels to use NEST indexing'
                a = np.right_shift(allpix, shift)
                c = np.unique(np.right_shift(conepix, shift))
                return np.isin(a,c)

            
            # a selection of all those in an outer cone
            incone = cone_select(allpix, conepix, 13)

            # times: convert to double, add to start, convert to MJD 
            time = MJD(np.array(table['time'],float)[incone]+tstart)
            in_gti = self.gti(time)
            if np.sum(in_gti)==0:
                print(f'WARNING: no photons for month {month}!')

            pixincone = allpix[incone][in_gti]
            
            # distance from center for all accepted photons
            ll,bb = healpy.pix2ang(self.nside, pixincone,  nest=self.nest, lonlat=True)
            cart = lambda l,b: healpy.dir2vec(l,b, lonlat=True) 
            t2 = np.degrees(np.array(np.sqrt((1.-np.dot(center, cart(ll,bb)))*2), np.float32)) 

            # assemble the DataFrame, remove those outside the radius
            out_df = pd.DataFrame(np.rec.fromarrays(
                [np.array(table['band'])[incone][in_gti], time[in_gti], pixincone, t2], 
                names='band time pixel radius'.split()))
            
            # apply final selection for radius and energy range
         
            if band_limits is None: return out_df.query(f'radius<{radius}')
            
            return out_df.query(f'radius<{radius} & {band_limits[0]} < band < {band_limits[1]}')

        # get the monthly-partitioned dataset and tstart values
        dataset = self.photon_data_source['dataset']
        tstart_dict= self.photon_data_source['tstart_dict']
        months = tstart_dict.keys() 

        if self.verbose>0: 
            print(f'Loading data from {len(months)} months from Arrow dataset {dataset}\n', end='')
       
        dflist=[] 
        for month in months:
            table= pq.read_table(dataset, filters=[f'month == {month}'.split()])
            tstart = tstart_dict[month]
            d = load_photon_data(table, tstart)
            if d is not None:
                dflist.append(d)
                if self.verbose>1: print('.', end='')
            else:
                if self.verbose>1: print('x', end='')
                continue

        assert len(dflist)>0, '\nNo photon data found?'
        df = pd.concat(dflist, ignore_index=True)
        if self.verbose>0:
            emin,emax = self.energy_range or (self.energy_edges[0],self.energy_edges[-1])
            print(f'\n\tSelected {len(df)} photons within {self.radius}'\
                  f' deg of  ({self.l:.2f},{self.b:.2f})')
            print(f'\tEnergies: {emin:.1f}-{emax:.0f} MeV')
            ta,tb = df.iloc[0].time, df.iloc[-1].time
            print(f'\tDates:    {UTC(ta):16} - {UTC(tb)}'\
                f'\n\tMJD  :    {ta:<16.1f} - {tb:<16.1f}')  
        return df  
        
        
def testdata(**kwargs):
    class Setup(object):
        def __init__(self, **d):
            self.__dict__.update(d)
    setup = Setup(source_name='Geminga', l=195.134, b=4.266, interval=1, mjd_range=None)
    setup.__dict__.update(**kwargs)
    return TimedData(setup)

#####################################################################################
#    Parquet code -- TODO just copied, need to test.
#           from notebooks/code development/parquet_writer.ipynb

class ParquetConversion(object):
    import glob, pickle;
    import healpy

    
    def __init__(self, 
                 data_file_pattern ='$FERMI/data/P8_P305/time_info/month_*.pkl',
            dataset = '/nfs/farm/g/glast/u/burnett/analysis/lat_timing/data/photon_dataset'):

        self.files = sorted(glob.glob(os.path.expandvars(data_file_pattern)));
        print(f'Found {len(self.files)} monthly files with pattern {data_file_pattern}'\
             f'\nWill store parquet files here: {dataset}')
        if os.path.exists(dataset):
            print(f'Dataset folder {dataset} exists')
        else:
            os.makedirs(dataset)
        self.dataset=dataset
            
    def convert_all(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        files=self.files
        dataset=self.dataset
        nside=1024
    
        def convert(month):

            infile = files[month-1]
            print(month, end=',')
            #print(f'Reading file {os.path.split(infile)[-1]} size {os.path.getsize(infile):,}' )   

            with open(infile, 'rb') as inp:
                t = pickle.load(inp,encoding='latin1')

            # convert to DataFrame, add month index as new column for partition, then make a Table
            df = pd.DataFrame(t['timerec'])
            tstart = t['tstart']
            df['month']= np.uint8(month)
            # add a columb with nest indexing -- makes the ring redundant, may remove later
            df['nest_index'] = healpy.ring2nest(nside, df.hpindex).astype(np.int32)
            table = pa.Table.from_pandas(df, preserve_index=False)

            # add to partitioned dataset
            pq.write_to_dataset(table, root_path=dataset, partition_cols=['month'] )

        for i in range(len(files)):
            convert(month=i+1)
