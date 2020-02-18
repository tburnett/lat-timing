"""
Process time data set
Expect to find set of files created by uw/data/timed_data/create_timed_data to generate files with times for all photons
Extract a single data set around a cone
"""

import os, pickle,  argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import keyword_options

from data_management import TimedData, TimedDataArrow
from weightman import WeightedData, WeightModel
from light_curve import LightCurve, BayesianBlocks



class Main(object):
    """Top-level processing for photon data
    """
 
    plt.rc('font', size=12)
    defaults=(
        ('verbose', 1, 'verbosity level'),
        ('radius',  5, 'cone radius for selection [deg]'),
        ('interval', 2, 'Binning time interval [days]'),
        ('mjd_range', None, 'Range of MJD: default all data'),
        ('weight_file', None, 'file name to find weight'),
        ('weight_model', None, 'file name for a weight model'),
        ('fix_weights',  True, 'Set to supplement weights with model' ),
        ('version',     2.0,   'version number: >1 is new data storage'),
       )
    
    @keyword_options.decorate(defaults)
    def __init__(self, name, position=None, **kwargs  ):
        """Set up combined data from set of monthly files

        name :    string, source name 
        position : an (l,b) pair [optional- if not present, use name to look up] 

        """
        keyword_options.process(self,kwargs)

        self._set_geometry(name, position)
        if self.version>1:
            self.timedata = TimedDataArrow(self, source_name=name, verbose=self.verbose)
        else:
            self.timedata = TimedData(self, source_name=name, verbose=self.verbose)

        if self.weight_file is not None:
            # adds weights from map and replace data object
            self.data = WeightedData(self.timedata, self.weight_file, self.fix_weights)
            if self.fix_weights:
                if self.verbose>0:
                    print(f'Creating weight model')
                df = self.data.photon_data
                weight_model = WeightModel.from_data(df, plotit=False)
                tofix = pd.isna(df.weight) | (df.band>13) 
                fixme= df.loc[tofix,:]
                fixed = fixme.apply(lambda c: weight_model(c.band, c.radius), axis=1,)  
                df.loc[tofix, 'weight'] = fixed
                self.wm = weight_model
            else:
                self.data= TimedData(self, source_name=name, verbose=self.verbose)
        elif self.weight_model is not None:
            self.data = WeightedData(self.timedata, weight_model=self.weight_model)
#             self.wm = weight_model = WeightModel(self.weight_model)
#             self.data = self.timedata.photon_data
        else:
            raise Exception('No weight processing specified')

    @property
    def photons(self):
        return self.df
    
    @property
    def cells(self, bins=None):
        """cells according to default binning, as a DataFrame
        """ 
        cells=dict()
        for i,cell in enumerate(self.data.binned_weights(None)):
            cells[i]= dict(time=cell['t'], n=cell['n'], w=np.array(cell['w']))
        return pd.DataFrame.from_dict(cells, orient='index'); 

    
    def _set_geometry(self, name, position):
        self.name=name
        if position is None:
            skycoord = SkyCoord.from_name(name)
            gal = skycoord.galactic
            self.l,self.b = (gal.l.value, gal.b.value)
        else:
            self.l,self.b = position
        if self.verbose>=0:
            print(f'\nSource "{self.name}" at: (l,b)=({self.l:.3f},{self.b:.3f}); ROI radius={self.radius}')
            
    def light_curve(self, bins=None, **kwargs):
        """ Rerurn a LIghtCurve object, containing a table of fluxes and other cell info
        bins: a list of bin edges | None
            if None use default
        """
        if bins is None and len(kwargs)==0: # set or use cached version unless new bins or a kwarg
            if not hasattr(self, 'basic_lc'):
                self.basic_lc = LightCurve(self.data.binned_weights(bins), **kwargs)
            return self.basic_lc
        return LightCurve(self.data.binned_weights(bins), **kwargs)

    def bayesian_blocks(self, lc=None, lc_kwargs={}, **kwargs):
        """
        lc : LightCurve object or None
            initial, presumably regular, binning to be used. If None, use default 
            bb_kwargs : dict
                parameters to pass to the BayesianBlocks class, especially fitness_func
            kwargs:
                parameter to psss for the light curve
        """
        bb = BayesianBlocks(lc or self.light_curve(), **kwargs)
        edges = bb.partition()
        lckw = dict(rep='poisson', min_exp=0.01)
        lckw.update(lc_kwargs)
        return bb.light_curve(edges, **lckw)
    

    def plot_time(self, radius_max=2, delta_t=2, xlim=None):
        """
        """
        df = self.df

        t = df.time.values
        ta,tb=t[0],t[-1]
        Nbins = int((tb-ta)/float(delta_t))

        fig,ax= plt.subplots(figsize=(15,5))
        hkw = dict(bins = np.linspace(ta,tb,Nbins), histtype='step')
        ax.hist(t, label='E>100 MeV', **hkw)
        ax.hist(t[(df.radius<radius_max) & (df.band>0)], label=f'radius<{radius_max} deg', **hkw)
        ax.set(xlabel=r'$\mathrm{MJD}$', ylabel='counts per {:.0f} day'.format(delta_t))
        if xlim is not None: ax.set(xlim=xlim)
        ax.legend()
        ax.set_title('{} counts vs. time'.format(self.name))

    def plot_radius(self, cumulative=False, squared=True):
        plt.rc('font', size=12)
        df = self.df
        fig,ax = plt.subplots(figsize=(6,3))
        x = df.radius**2 if squared else df.radius
        hkw = dict(bins=np.linspace(0, 25 if squared else 5, 100), 
                   histtype='step',lw=2,cumulative=cumulative)
        ax.hist(x, label='E>100 MeV', **hkw)
        ax.hist(x[df.band>8], label='E>1 GeV', **hkw)
        ax.set(yscale='log', xlabel='radius**2 [deg^2]' if squared else 'delta [deg]', 
            ylabel='cumulative counts' if cumulative else 'counts'); 
        ax.legend(loc='upper left' if cumulative else 'upper right');


### Code that must be run in FermiTools context to create the database
#from uw/data import binned_data
# def create_timed_data(
#         monthly_ft1_files='/afs/slac/g/glast/groups/catalog/P8_P305/zmax105/*.fits',
#         outfolder='$FERMI/data/P8_P305/time_info/',
#         overwrite=False  ):
#     """
#     """
#     files=sorted(glob.glob(monthly_ft1_files))
#     assert len(files)>0, 'No ft1 files found at {}'.format(monthly_ft1_files)
#     gbtotal = np.array([os.stat(filename).st_size for filename in files]).sum()/2**30
#     print '{} FT1 files found, {} GB total'.format(len(files), gbtotal)
#     outfolder = os.path.expandvars(outfolder)
#     if not os.path.exists(outfolder):
#         os.makedirs(outfolder)
#     os.chdir(outfolder)   
#     for filename in files:
#         m = filename.split('_')[-2]
#         outfile = 'month_{}.pkl'.format(m)
#         if not overwrite and os.path.exists(outfile) :
#             print 'exists: {}'.format(outfile)
#             continue
#         print 'writing {}'.format(outfile),
#         tr = binned_data.ConvertFT1(filename).time_record()
#         pickle.dump(tr, open(outfile, 'w'))

class CombinedLightcurves(object):
    
    defaults = Main.defaults
  
    @keyword_options.decorate(defaults)
    def __init__(self, sources, path=None, image_dir=None, **kwargs):
        """Generate one or many light curves with the BB version overlayed on an interval
        
        sources : list of source dicts or a single one
            the dict must have at least a 'name' key
            optional: 
                l,b, or ra, dec--the position will be determined if the name is recognized
                weight_model or weight_file--use default if not present
        """
        keyword_options.process(self,kwargs)
        if isinstance(sources, dict):
            sources=[sources]
        
        def position(s):
            keys = list(s.keys())
            if 'l' in keys and 'b' in keys:
                return (s['l'], s['b'])
            if 'ra' in keys and 'dec' in keys:
                s = SkyCoord(s['ra'],s['dec'], unit='deg', frame='fk5').transform_to('galactic')
                return( s.l.value, s.b.value)
            return None
        
        for s in sources:
            try:

                self.source_name=s['name']
                self.cdata = Main(s['name'], position=position(s), 
                                  mjd_range=self.mjd_range, 
                                  weight_model= s.get('weight_model', self.weight_model),
                                  weight_file = s.get('weight_file', self.weight_file), 
                                  verbose=self.verbose,
                                  version=self.version)

            except Exception as msg:
                print(f'Failed: {msg}')

            lc = self.cdata.light_curve()
            self.edges = lc.data.edges
            self.rep = lc.rep
            self.lc_df = lc.fit_df
            bb = self.cdata.bayesian_blocks(fitness_func='likelihood')
            self.bb_df = bb.fit_df
            if path is not None:
                self.write(path)
            if image_dir is not None:
                self.flux_plots(outdir=image_dir)
           
    @classmethod
    def from_file(cls, filename , verbose=1):

        with open(filename, 'rb') as inp:
            d = pickle.load(inp)
            self = cls([])
            self.verbose = verbose
            self.source_name = d['source_name']
            self.lc_df = pd.DataFrame.from_dict(d['lc_dict'])
            self.bb_df = pd.DataFrame.from_dict(d['bb_dict'])
            return self
    
    @classmethod
    def from_fits(cls, source_name, lc, bb):
        self = cls([])
        self.source_name=source_name
        self.lc_df = lc.fit_df if lc is not None else None
        self.bb_df = bb.fit_df if bb is not None else None
        return self

    def write(self, path='.'):

        filename= self.source_name.replace(' ','_').replace('+','p')
        fullfn = f'{path}/{filename}.pkl'
        with open(fullfn, 'wb') as out:
            pickle.dump(
                dict(source_name=self.source_name, 
                        rep=self.rep,
                        edges = self.edges,
                        lc_dict = self.lc_df.to_dict('records'),
                        bb_dict = self.bb_df.to_dict('records')
                    ),
                out)
        print(f'Wrote light curve and Bayesian blocks for source "{self.source_name}" to\n   {fullfn}')

    
    def flux_plots(self, ax=None, outdir=None, **kwargs):
        import matplotlib.ticker as ticker
            
        def flux_plot( df, ts_max=9,  step=False, **kwargs): 
            """Make a plot of flux assuming Poisson rep info
            df : DataFrame generated by LightCurve
            ts_max : error bar above this
            step : bool
                if True, make steps
            """
            if df is None: return
            nd = self.interval
            kw=dict(yscale='log', xlabel='MJD', 
                    ylabel=f'relative flux per {"day" if nd==1 else str(nd)+" days"}',
                    color='blue', lw=1, fmt='+')
            kw.update(**kwargs)

            lw= kw.pop('lw')
            yscale = kw.pop('yscale')
            ax.grid(alpha=0.5)
            ax.set(ylabel=kw.pop('ylabel'), xlabel=kw.pop('xlabel'), yscale=yscale)

            ts = df.ts
            limit = ts<ts_max
            bar = df.loc[~limit,:]
            lim = df.loc[limit,:]
            
            if step:
                t =df.t.values
                xerr= df.tw.values/2
                y =np.select([~limit, limit], [df.flux, df.limit])

                dy=[bar.errors.apply(lambda x: x[i]).clip(0,4) for i in range(2)]
                ax.step( np.append(t-xerr,[t[-1]+xerr[-1]]), 
                         np.append(y,     [y[-1]]         ), color=kw['color'], where='post') 
                ax.errorbar(x=bar.t, y=bar.flux, yerr=dy,  elinewidth=lw, **kw)
                ax.errorbar(x=lim.t, y=lim.limit, xerr=None, yerr=0.2*lim.limit,
                            uplims=True, ls='', lw=2, capsize=6, capthick=0, **kw)
                return



            # the points with error bars
            t = bar.t
            xerr = bar.tw/2 
            y =  bar.flux.values
            dy = [bar.errors.apply(lambda x: x[i]).clip(0,4) for i in range(2)]
            ax.errorbar(x=t, y=y, xerr=xerr if not step else None, yerr=dy,  elinewidth=lw, **kw)

            # now do the limits
            if len(lim)>0:
                t = lim.t
                xerr = lim.tw/2
                y = lim.limit.values
                yerr=0.1*(1 if yscale=='linear' else y)
                ax.errorbar(x=t, y=y, xerr=xerr,
                        yerr=yerr,  color='wheat', 
                        uplims=True, ls='', lw=2, capsize=4, capthick=0,
                        alpha=0.5)


        xlim = kwargs.pop('xlim', None)
        ylim = kwargs.pop('ylim', None)
        fig, ax = plt.subplots(figsize=(15,4)) if ax is None else (ax.figure, ax)

        flux_plot(self.lc_df,  color='lightgrey', lw=1, fmt=' ')
        flux_plot(self.bb_df,  color='blue',lw=1, fmt='.', step=True)
        ax.set(xlim=xlim, ylim=ylim)
        ax.text(0.02, 0.85, self.source_name, transform=ax.transAxes)

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda val,pos: { 0.1:'0.1',1.0:'1', 10.0:'10', 100.:'100'}.get(val,'')))
        if outdir is not None:
            filename = self.source_name.replace(' ', '_',).replace('+','p')+'.png'
            fig.tight_layout()
            fig.savefig(f'{outdir}/{filename}')
            if self.verbose>0:
                print(f'Saved figure to {filename}')
        
    def to_galactic(ra,dec):
        """return (l,b) given ra,dec"""
        from astropy.coordinates import SkyCoord
        g = SkyCoord(ra, dec, unit='deg', frame='fk5').galactic
        return g.l.value, g.b.value
        
    
    def generate_light_curves(fn =  '/nfs/farm/g/glast/g/catalog/pointlike/skymodels/P8_10years/uw9010/plots/associations/bzcat_summary.csv', 
                              lcpath = '/nfs/farm/g/glast/u/burnett/analysis/lat_timing/data/light_curves/'):
        df = pd.read_csv(fn, index_col=0)

        for name, rec in df.iterrows():
            if name.startswith('5B'):
                name = name[:4]+' '+name[4:]
            print(f'\n======={name}, TS={rec.ts:.0f}')
            l,b = to_galactic(rec.ra, rec.dec)

            u = main.CombinedLightcurves([name], [(l,b)], path=lcpath, verbose=0)
        
    if __name__=='__main__':
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Create light curves")
        parser.add_argument('input',  nargs='?',  
                            help='csv file name with source info')
        parser.add_argument('output_path', nargs='?', default='.', help='path to save output files')
#         parser.add_argument('-p', '--proc', 
#                         default=, 
#                         help='proc name,  default: "%(default)s"')
        
        parser.print_help()
        args = parser.parse_args()
        print(args.input, args.output_path)
#         generate_light_curves()