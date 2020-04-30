import os,inspect
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np

def formatter(funct, **kwargs):
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
        
    Finally runs the IPython display.
    
    funct : the caller function
        
    **kwargs
        additional values to formatted in the docstring
    """    
    
    # use inspect to get caller frame, the function name, and locals dict
    assert inspect.isfunction(funct), f'Expected a function'
    doc = inspect.getdoc(funct)
    back =inspect.currentframe().f_back
    name= inspect.getframeinfo(back).function
    locs = inspect.getargvalues(back).locals
    locs.update(kwargs) # add kwargs
    
    # setup path to save figures in folder with function name
    path = f'figs/{name}/'
    os.makedirs(path, exist_ok=True)
    
    # process each figure found in locals: assume we want to display them 
    def process_figure(fig):
        n = fig.number
        caption=getattr(fig,'caption', '').format(**locs)
        fn = f'{path}fig_{n}.png'
        # save the figure, include a link in text
        fig.savefig(fn)
        plt.close(fig) # so IPython does not display it if inline set
        # add the HTML to insert the image, allowing for a caption, to the object
        html =  f'<figure> <img src="{fn}" alt="Figure {n}">'\
                f' <figcaption>{caption}</figcaption>'\
                '</figure>'
        fig.html=html
        
    for key,value in locs.items():
        if isinstance(value, plt.Figure):
            process_figure(value)
    
    # apply locals, with kwargs, dict to the doc, and pass it to IPython's display as markdown
    display.display(display.Markdown(doc.format(**locs)))

def demo_function( xlim=(0,10)):
    """
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
    Check value of the kwarg *test* passed to the formatter: it is "{test}".
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
    