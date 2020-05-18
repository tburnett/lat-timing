"""Generate documents for Jupyter display 
"""

import os, inspect, string, io, datetime
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def doc_display(funct, folder_path='figs', fig_kwargs={}, df_kwargs={}, **kwargs):
    """Format and display the docstring, as an alternative to matplotlib inline in jupyter notebooks
    
    Parameters
    ----------    
    funct : function object | dict
        if a dict, has keys name, doc, locs
        otherlwise, the calling function, used to obtain is name, the docstring, and locals
    folder_path : string, optional, default 'figs'
    fig_kwargs : dict,  optional
        additional kwargs to pass to the savefig call.
    df_kwargs : dict, optional, default {"float_format":lambda x: f'{x:.3f}', "notebook":True, "max_rows":10, "show_dimensions":}
        additional kwargs to pass to the to_html call for a DataFrame
    kwargs : used only to set variables referenced in the docstring.    
    
    Expect this to be called from a function or class method that may generate one or more figures.
    It takes the function's docstring, assumed to be in markdown, and applies format() to format values of any locals. 
    
    Each Figure to be displayed must have a reference in the local namespace, say `fig`, and have a unique number. 
    Then the  figure will be rendered at the position of a '{fig}' entry.
    In addition, if an attribute "caption" is found in a Figure object, its text will be displayed as a caption.
    
    Similarly, if there is a reference to a pandas DataFrame, say a local variable `df`, then any occurrence of `{df}`
    will be replaced with an HTML table, truncated according to the notebook format.
    
    A possibly important detail, given that The markdown processor expects key symbols, like #, to be the first on a line:
    Docstring text is processed by inspect.cleandoc, which cleans up indentation:
        All leading whitespace is removed from the first line. Any leading whitespace that can be uniformly
        removed from the second line onwards is removed. Empty lines at the beginning and end are subsequently
        removed. Also, all tabs are expanded to spaces.
        
    Finally runs the IPython display to process the markdown for insertion after the code cell that invokes the function.
   
    Unrecognized entries are ignored, allowing latex expressions. (The string must be preceded by an "r"). In case of
    confusion, double the curly brackets.

    """  
    # check kwargs
    dfkw = dict(float_format=lambda x: f'{x:.3f}', notebook=True, max_rows=10, show_dimensions=False, justify='right')
    dfkw.update(df_kwargs)

    # get docstring from function object
    expected_keys='doc name locs'.split()
    if inspect.isfunction(funct): #, f'Expected a function: got {funct}'
        doc = inspect.getdoc(funct)

        # use inspect to get caller frame, the function name, and locals dict
        back =inspect.currentframe().f_back
        name= inspect.getframeinfo(back).function
        locs = inspect.getargvalues(back).locals.copy() # since may modify

    elif (type(funct) == dict) and (set(expected_keys) == set(funct.keys())):
        doc=funct['doc']
        name=funct['name']
        locs = funct['locs'].copy()
    else:
        raise Exception(f'Expected a function or a dict with keys {expected_keys}: got {funct}')
    
    # add kwargs if any
    locs.update(kwargs)
    
    # set up path to save figures in folder with function name
    path = f'{folder_path}/{name}'

    
    # process each Figure or DataFrame found in local for display 
    
    class FigureWrapper(plt.Figure):
        def __init__(self, fig):
            self.__dict__.update(fig.__dict__)
            self.fig = fig
            
        @property
        def html(self):
            # backwards compatibility with previous version
            return self.__str__()
            
        def __str__(self):
            if not hasattr(self, '_html'):
                fig=self.fig
                n = fig.number
                caption=getattr(fig,'caption', '').format(**locs)
                # save the figure to a file, then close it
                fig.tight_layout(pad=1.05)
                fn = f'{path}/fig_{n}.png'
                fig.savefig(fn) #, **fig_kwargs)
                plt.close(fig) 

                # add the HTML as an attribute, to insert the image, including optional caption

                self._html =  f'<figure> <img src="{fn}" alt="Figure {n}">'\
                        f' <figcaption>{caption}</figcaption>'\
                        '</figure>\n'
            return self._html
        
        def __repr__(self):
            return self.__str__()

        
    class DataFrameWrapper(object): #pd.DataFrame):
        def __init__(self, df):
            #self.__dict__.update(df.__dict__) #fails?
            self._df = df
        @property
        def html(self):
            # backwards compatibility with previous version
            return self.__str__()
        def __repr__(self):
            return self.__str__()
        def __str__(self):
            if not hasattr(self, '_html'):
                self._html = self._df.to_html(**dfkw) # self._df._repr_html_()                
            return self._html

            
    def figure_html(fig):
        if hasattr(fig, 'html'): return
        os.makedirs(path, exist_ok=True)
        
        return FigureWrapper(fig)
        
    def dataframe_html(df):
        if hasattr(df, 'html'): return None
        return DataFrameWrapper(df)
   
    def processor(key, value):
        # value: an object reference to be processed 
        ptable = {plt.Figure: figure_html,
                  pd.DataFrame: dataframe_html,
                 }
        f = ptable.get(value.__class__, lambda x: None)
        # process the reference: if recognized, there may be a new object
        newvalue = f(value)
        if newvalue is not None: 
            locs[key] = newvalue
            #print(f'key={key}, from {value.__class__.__name__} to  {newvalue.__class__.__name__}')
    
    for key,value in locs.items():
        processor(key,value)
   
    # format local references. Process Figure or DataFrame objects found to include .html representations.
    # Use a string.Formatter subclass to ignore bracketed names that are not found
    #adapted from  https://stackoverflow.com/questions/3536303/python-string-format-suppress-silent-keyerror-indexerror

    class Formatter(string.Formatter):
        class Unformatted:
            def __init__(self, key):
                self.key = key
            def format(self, format_spec):
                return "{{{}{}}}".format(self.key, ":" + format_spec if format_spec else "")

        def vformat(self, format_string,  kwargs):
            try:
                return super().vformat(format_string, [], kwargs)
            except AttributeError as msg:
                return f'Failed processing because: {msg.args[0]}'
        def get_value(self, key, args, kwargs):
            return kwargs.get(key, Formatter.Unformatted(key))

        def format_field(self, value, format_spec):
            if isinstance(value, Formatter.Unformatted):
                return value.format(format_spec)
            #print(f'\tformatting {value} with spec {format_spec}') #', object of class {eval(value).__class__}')
            return format(value, format_spec)
                        
    docx = Formatter().vformat(doc+'\n', locs)       
    # replaced: docx = doc.format(**locs)

    # pass to IPython's display as markdown
    display.display(display.Markdown(docx))
    
    return docx


def markdown(text):
    """Add text to the display"""
    display.display(display.Markdown(text))
    
def md_to_html(output, filename):
    """write nbconverted markdown to a file 
    
    parameters
    ----------
    output : string | IPython.utils.capture.CapturedIO object
        if not a string extract the markdown from each of the outputs list 
    """
    from nbconvert.exporters import  HTMLExporter
       
    if type(output)==str:
        md_text=output
    elif hasattr(output, 'outputs'):
        md_text=''
        for t in output.outputs:            
            md_text += '\n\n'+t.data['text/markdown']
    else:
        raise Exception(f'output not recognized: {output.__class__} not a string or CapturedIO object?')
    
    class Dict(dict):
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.update(kwargs)
    nb = Dict(
            cells= [Dict(cell_type="markdown", 
                         metadata={}, 
                         source=md_text,
                        )
                   ],
            metadata={},
            nbformat=4,
            nbformat_minor=4,
            )

    # now pass it to nbconvert to write as an HTML file
    exporter = HTMLExporter()
    output, resources = exporter.from_notebook_node(nb) 
    with open(filename, 'wb') as f:
        f.write(output.encode('utf8'))

class Displayer(object):
    """Base class for display purposes
    A subclass must run super().__init__(). Then any member function that calls self.display()
    will have its docstring processed.
    
    Implements the with construction. 
   
    """
    def __init__(self, path=None, fignum=1, html_file=None, pdf_file=None):
        """
        path : None or string
            Path to save figures
            if None, use figs/<classname>
        fignum : optional, default 1
            First figure number to use
        html_file : optional, default None
            if set, save the accumulated output to an HTML file when closed
        pdf_file : optional default None
            If set, save the accumulated docstring output to the PDF file when closed.
        """
        self.path=path or f'figs/{self.__class__.__name__}'
        os.makedirs(self.path, exist_ok=True)
        self._fignum=fignum-1

        self.date=str(datetime.datetime.now())[:16]
        self.pdf_file=pdf_file
        self.html_file=html_file
        self.data = ''
        
    def __enter__(self): return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.pdf_file:
            print('*** PDF generation not yet implemented!')
        if self.html_file:
            md_to_html(self.data, self.html_file )
#             self.markdown(f'---\nDocument saved to {self.html_file}')

            
    def markdown(self, text, clean=True):
        """Add md text to the display"""
        if clean:
            text= inspect.cleandoc(text)
        # make sure clean separation before and after
        display.display(display.Markdown(text)) 
        self.data += '\n\n'+ text
                        
    
    def footer(self, source_file):
        # This is Fermi-LAT specific, assuming run at SLAC
        try:
            from __init__ import repository_name, SLAC_path, github_path
        except:
            return '*footer expects defining stuff in local __init__.py!*'
        doc_path = '' 
        if self.html_file:
            curpath = os.getcwd()
            rep_path, rep_name = os.path.split(SLAC_path)
            i = curpath.find(rep_name)
            j =curpath.find('/')+i
            if i==-1 or j==-1:
                doc_path = f'Problem construcing SLAC doc'
            else:
                rep_path+curpath[j-1:]+'/'+self.html_file+'?skipDecoration'
                doc_path= f'[Document at SLAC (Fermi access)]({rep_path+curpath[j-1:]}/{self.html_file}?skipDecoration)'        
        return self.markdown(
            f"""\
            ---
            This code, `{source_file}`, is part of my repository `{repository_name}`,
            and can be found at [github]({github_path}/{source_file})
            or, the current version (Fermi-LAT access) at [SLAC]({SLAC_path}/{source_file}?skipDecoration).
            
            (Document created using [`docstring.py`]({github_path}/code/docstring.py).
            
            {doc_path}
            """
            )                       

    def newfignum(self):
        self._fignum+=1
        return self._fignum
    
    def display(self,  **kwargs):                
        # use inspect to get caller frame, the function name, and locals dict
        back =inspect.currentframe().f_back
        name= inspect.getframeinfo(back).function
        locs = inspect.getargvalues(back).locals
        
        # construct the calling function object to get its docstring
        funct =eval(f'self.{name}')
        doc = inspect.getdoc(funct)
        
        #  
        md_data = doc_display(dict(name=name, doc=doc, locs=locs), 
                    folder_path=self.path, **kwargs)
        self.data += '\n\n'+md_data
        
def demo_function( xlim=(0,10)):
    r"""
    ### Function generating figures and table output

    Note the value of the arg `xlim = {xlim}`

    * Display head of the dataframe used to make the plots
    {dfhead}
    
    * **Figure 1.**  
    Describe analysis for this figure here.
    {fig1}
    Interpret results for Fig. {fig1.number}.  
    
    * **Figure 2.**  
    A second figure!
    {fig2}
    This figure plots a square root

    ---
    Check value of the kwarg *test* passed to the formatter: it is "{test:.2f}".
    
    ---
    Insert some latex to test that it passes unrecognized curly bracket entries on...and that they get rendered!
    <br>
        \begin{align*}
        \sin^2\theta + \cos^2\theta =1
        \end{align*}
    An inline formula: $\frac{1}{2}=0.5$
    """
    plt.rc('font', size=14)
    x=np.linspace(*xlim)
    df= pd.DataFrame([x,x**2,np.sqrt(x)], index='x xx sqrtx'.split()).T
    dfhead =df.head(2)
    
    fig1,ax=plt.subplots(num=1, figsize=(4,3))
    ax.plot(df.x, df.xx)
    ax.set(xlabel='$x$', ylabel='$x^2$', title=f'figure {fig1.number}')
    fig1.caption="""Example caption for Fig. {fig1.number}, which
            shows $x^2$ vs. $x$.    
            <p>A second caption line."""
    
    fig2,ax=plt.subplots(num=2, figsize=(4,3))
    fig2.caption='A simple caption.'
    ax.set_title('figure 2')
    ax.plot(df.x, df.sqrtx)
    ax.set(xlabel = '$x$', ylabel=r'$\sqrt{x}$')      
    
    doc_display(demo_function, test=99)
    
    
class DemoClass(Displayer):
    def __init__(self):
        super().__init__()
        
    def demo(self):
        r"""### Formatting summary
        
        This is generated using a member function of a subclass of `docstring.Displayer`.
        There are three elements in such a function: the docstring, written in 
        markdown, containing 
        local names in curly-brackets, the code, and a final call `self.display()`.  
        Unlike a formatted string, entries in curly brackets cannot be expressions.
        
        #### Local variables  
        Any variable, say from an expression in the code `q=1/3`, can be interpreted
        with `"q={{q:.3f}}"`, resulting in  
        q={q:.3f}.
    
        
        #### Figures
        
        Any number of Matplotlib Figure objects can be added, with unique numbers.
        An entry "{{fig1}}", along with code that defines `fig1` like this:
        ```
        x = np.linspace(0,10)
        fig1,ax=plt.subplots(num=self.newfignum(), figsize=(4,3))
        ax.plot(x, x**2)
        ax.set(xlabel='$x$', ylabel='$x^2$', title=f'figure {{fig1.number}}')
        fig1.caption='''Caption for Fig. {fig1.number}, which
                shows $x^2$ vs. $x$.'''
        ```
        produces:
        
        {fig1}
        
        The display processing replaces the `fig1` reference in a copy of `locals()`
        with a object that implements a `__str__()` function returning an HTML reference 
        to a saved png representation of the figure.   
        Note the convenience of defining a caption by adding the text as an attribute of
        the Figure object.  
        Also, the `self.newfignum()` returns a new figure number.
        
        #### DataFrames
        A `DataFrame` object is treated similarly. 
        
        The code
        ```
        df = pd.DataFrame.from_dict(dict(x=x, xx=x**2)).T
        
        ```
        
        <br>Results in "{{df}}" being replaced with
        {df}
    
        #### LaTex 
        Jupyter-style markdown can contain LaTex expressions. For this, it is 
        necessary that the docstring be preceded by an "r". So,
        ```
        \begin{{align*}}
        \sin^2\theta + \cos^2\theta =1
        \end{{align*}}
        ```
        
        <br>Results in:   
            \begin{align*}
            \sin^2\theta + \cos^2\theta =1
            \end{align*}
        
        ---
        """
        q = 1/3
        xlim=(0,10)
        plt.rc('font', size=14)
        x=np.linspace(*xlim)
        df= pd.DataFrame([x,x**2,np.sqrt(x)], index='x xx sqrtx'.split()).T
        dfhead =df.head(2)

        fig1,ax=plt.subplots(num=self.newfignum(), figsize=(4,3))
        ax.plot(x, x**2)
        ax.set(xlabel='$x$', ylabel='$x^2$', title=f'figure {fig1.number}')
        fig1.caption="""Caption for Fig. {fig1.number}, which
                shows $x^2$ vs. $x$.    
               ."""
        
        df = pd.DataFrame.from_dict(dict(x=x, xx=x**2))
        dfhead = df.head()
        
        self.display()