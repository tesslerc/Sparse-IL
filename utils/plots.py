import numpy as np
import logging


def vis_plot(viz, plot_name, values):
    if viz is not None:
        plot_data = np.array(values)
        viz.line(X=plot_data[:, 0], Y=plot_data[:, 1], win=plot_name, opts=dict(title=plot_name))


def viz(visualize, vis_name):
    if not visualize:
        vis = None
    else:
        from visdom import Visdom
        vis = Visdom(env=vis_name)
        logging.info('To view results, run \'python -m visdom.server\'')  # activate visdom server on bash
        logging.info('then head over to http://localhost:8097')  # open this address on browser

    return vis
