import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

fig, axes = plt.subplots(4, sharey = False) # create figure
plots = [ax.plot([])[0] for ax in axes]

fig.tight_layout()
fig.subplots_adjust(bottom=0.12)

t1 = np.arange(0.0, 5.0, 0.1)

def update(idx):

    a1 = np.sin(idx*np.pi *t1)
    a2 = np.sin((idx/2)*np.pi*t1)
    a3 = np.sin((idx/4)*np.pi*t1)
    a4 = np.sin((idx/8)*np.pi*t1)

    for plot, a in zip(plots, [a1,a2,a3,a4]):
        plot.set_data(t1, a)
        plot.axes.relim()
        plot.axes.autoscale()

    fig.canvas.draw_idle()

update(5)
slider = Slider(fig.add_axes([.1,.04,.6,.03]), "Label", 0,10,5)
slider.on_changed(update)
plt.show()