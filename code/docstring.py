import os,inspect
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np

def formatter(obj, **kwargs):
    """Expect to be called from a class method that generates one or more figures.
    Takes the function's markdown docstring, and applies format to it. Each figure should have a 
    corresponding {fig} entry.
    
    obj : object 
        get class name from this
    **kwargs
        values to set, if found in the docstring
    """    
        
    # use inspect to get caller frame, the function name, and locals dict
    back =inspect.currentframe().f_back
    name= inspect.getframeinfo(back).function
    locs = inspect.getargvalues(back).locals
    # setup path to save figures
    path = f'figs/{name}/'
    os.makedirs(path, exist_ok=True)

    for key,value in locs.items():
        if isinstance(value, plt.Figure):
            # change the value to include the image of the figure
            n = value.number
            fn = f'{path}fig_{n}.png'
            # save the figure, include a link in text
            value.savefig(fn)
            locs[key] = f'\n![Fig. {n}]({fn} "{n}")'
            #locs[key] = f'<img src="{fn}"/>'
            
    # update or add from kwargs
    locs.update(kwargs)
    
    # get the caller dostring, return formatted version
    doc = eval('obj.'+name + '.__doc__')
    plt.close('all') # so IPython doesn't display
    out=(doc.format(**locs))
    display.display(display.Markdown(out))

class Demo():
    
    def __init__(self, xlim=(0,10)):
        self.xlim=xlim
        
    def plots(self):

        """## Analysis demo

        \nNote that `xlim = {self.xlim}`

        \n#### Figure 1
        {fig1}<br>This figure is a square
        \n#### Figure 2
        {fig2}
        <br>This figure is a sqrt

        Check value of test: {test}
        """

        fig1,ax=plt.subplots(num=1, figsize=(4,4))
        ax.set_title('figure 1')
        x=np.linspace(*self.xlim)
        ax.plot(x, x**2)

        fig2,ax=plt.subplots(num=2, figsize=(4,4))
        ax.set_title('figure 2')
        ax.plot(x, np.sqrt(x))
        formatter(self, test=99)
    
