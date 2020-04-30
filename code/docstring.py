import os,inspect,string
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np

def formatter(funct, folder_path='figs', **kwargs):
    """Docstring formatter as an alternative to matplotlib inline in jupyter notebooks
    
    Expect to be called from a function or class method that generates one or more figures.
    Takes the function's docstring, assumed to be in markdown, and applies format() to format values of any locals. 
    Each figure, a plt.Figure object, should have a corresponding {fig.html} entry.
    During processing, the figure is saved to a local file, and an attribute "html" added to it.
    If an attribute "caption" is found in a Figure object, the text will be displayed as a caption.
    
    A possibly important detail, given that The markdown processor expects key symbols, like #, to be the first on a line.
    Docstring text is processed by inspect.cleandoc, which cleans up indentation:
        All leading whitespace is removed from the first line. Any leading whitespace that can be uniformly
        removed from the second line onwards is removed. Empty lines at the beginning and end are subsequently
        removed. Also, all tabs are expanded to spaces.
        
    Finally runs the IPython display to process the markdown for insertion after the code cell that invokes the function.
   
    Unrecognized entries are ignored, allowing latex expressions. (The string must be preceded by an "n"). In case of
    confusion, double the curly brackets.
    
    funct : the caller function
    folder_path : string
        where to put figures
    **kwargs
        additional values to formatted in the docstring
    """    
    
    # get docstring from function object
    assert inspect.isfunction(funct), f'Expected a function: got {funct}'
    doc = inspect.getdoc(funct)
        
    # use inspect to get caller frame, the function name, and locals dict
    back =inspect.currentframe().f_back
    name= inspect.getframeinfo(back).function
    locs = inspect.getargvalues(back).locals
    locs.update(kwargs) # add kwargs
    
    # set up path to save figures in folder with function name
    path = f'{folder_path}/{name}'
    os.makedirs(path, exist_ok=True)
    
    # process each figure found in local for display 
    def process_figure(fig):
        n = fig.number
        caption=getattr(fig,'caption', '').format(**locs)
        fn = f'{path}/fig_{n}.png'
        # save the figure, include a link in text
        fig.savefig(fn)
        plt.close(fig) # so IPython does not display it if inline set
        # add the HTML to insert the image, including optional caption
        html =  f'<figure> <img src="{fn}" alt="Figure {n}">'\
                f' <figcaption>{caption}</figcaption>'\
                '</figure>'
        fig.html=html
        
    [process_figure(value) for value in locs.values() if isinstance(value, plt.Figure)]
    
    # format local references, including figure HTML,
    # Use a string.Formatter subclass to ignore bracketed names that are not found
    #adapted from  https://stackoverflow.com/questions/3536303/python-string-format-suppress-silent-keyerror-indexerror

    class Formatter(string.Formatter):
        class Unformatted:
            def __init__(self, key):
                self.key = key
            def format(self, format_spec):
                return "{{{}{}}}".format(self.key, ":" + format_spec if format_spec else "")

        def vformat(self, format_string,  kwargs):
            return super().vformat(format_string, [], kwargs)
        def get_value(self, key, args, kwargs):
            try:
                return kwargs[key]
            except KeyError:
                return Formatter.Unformatted(key)
        def format_field(self, value, format_spec):
            return   value.format(format_spec) if isinstance(value, Formatter.Unformatted)\
                else format(value, format_spec)

    docx = Formatter().vformat(doc, locs)       
    # replaced: docx = doc.format(**locs)

    # pass to IPython's display as markdown
    display.display(display.Markdown(docx))

def demo_function( xlim=(0,10)):
    r"""
    ## Analysis demo

    Note that `xlim = {xlim}`

    #### Figure 1
    Describe analysis for this figure here.
    {fig1.html}
    Interpret results for Fig. {fig1.number}.
    
    #### Figure 2
    A second figure!
    {fig2.html}
    This figure is a sqrt
    
    ---
    Check value of the kwarg *test* passed to the formatter: it is "{test:.2f}".
    
    ---
    Insert some latex to test that it passes unrecognized entries on.
        \begin{align*}
        \sin(\theta)^2 + \cos(\theta)^2 =1
        \end{align*}
    An inline formula: $\frac{1}{2}=0.5$
    """

    x=np.linspace(*xlim)
    fig1,ax=plt.subplots(num=1, figsize=(4,4), tight_layout=True)
    ax.set_title('figure 1')
    ax.plot(x, x**2)
    fig1.caption="""Example caption for Fig. {fig1.number}, which
            is a square.
            <p>A second caption line."""
    
    fig2,ax=plt.subplots(num=2, figsize=(4,4))
    ax.set_title('figure 2')
    ax.plot(x, np.sqrt(x))
    
    formatter(demo_function, test=99)
    