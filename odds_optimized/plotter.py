import math

import matplotlib.pyplot as plt
import numpy as np

from .base import BaseDetector, BasePlotter


class LevelsetPlotter(BasePlotter):
    """
    Plotter for levelsets associated with a model.

    Attributes
    ----------
    model: BaseDetector
        the model to study

    Methods
    -------
    See BasePlotter methods.

    """

    def __init__(self, model: BaseDetector):
        assert model.__dict__.get("p") is not None
        assert model.__dict__["p"] == 2
        self.model = model

    def plot(self, x, *args, n_x1=100, n_x2=100, levels:list=None, percentiles:list=None, show=True, save=False, save_title="CF_levelset.png", close=True, **kwargs):
        """
        Plot the level sets of the model's decision function.
        The reference level {n+d choose d} is plotted in red.

        Parameters
        ----------
        x: np.ndarray
            The data points to plot, shape (n_samples, n_features=2).
        n_x1: int
            Number of points along the first dimension for the grid.
        n_x2: int
            Number of points along the second dimension for the grid.
        levels: list, optional
            Specific levels to plot. If None, defaults to [].
        percentiles: list, optional
            Percentiles to compute and plot as levels. If None, defaults to [].
        show: bool
            Whether to show the plot. Defaults to True.
        save: bool
            Whether to save the plot to a file. Defaults to False.
        save_title: str
            Title for the saved plot file. Defaults to "CF_levelset.png".
        close: bool
            Whether to close the plot or return it after saving or showing. Defaults to True.
        """
        assert x.shape[1] == self.model.__dict__["p"]
        # Make figure
        fig, ax = plt.subplots()
        
        # Scatter the data points
        ax.scatter(x[:, 0], x[:, 1], marker='x', s=20)

        # Make a grid and compute the function values
        x1_margin = (np.max(x[:, 0]) - np.min(x[:, 0])) / 5
        x2_margin = (np.max(x[:, 1]) - np.min(x[:, 1])) / 5
        ax.set_xlim([np.min(x[:, 0]) - x1_margin, np.max(x[:, 0]) + x1_margin])
        ax.set_ylim([np.min(x[:, 1]) - x2_margin, np.max(x[:, 1]) + x2_margin])
        X1 = np.linspace(np.min(x[:, 0] - x1_margin), np.max(x[:, 0] + x1_margin), n_x1)
        X2 = np.linspace(np.min(x[:, 1]) - x2_margin, np.max(x[:, 1] + x2_margin), n_x2)
        x1, x2 = np.meshgrid(X1, X2)
        x3 = np.c_[x1.ravel(), x2.ravel()]
        X3 = self.model.score_samples(x3).reshape(x1.shape)

        # Set default percentiles and levels
        if percentiles is None:
            percentiles = []
        if levels is None:
            levels = []
        levels += [np.percentile(X3, p) for p in percentiles]
        levels = sorted(set(levels))
        # Plot level sets
        cs = ax.contour(X1, X2, X3, levels=levels)
        ax.clabel(cs, inline=1)

        # Compute reference level to plot
        reference_level = math.comb(self.model.__dict__.get("p") + self.model.__dict__.get("d"), self.model.__dict__.get("p"))
        # Plot the reference level set
        cs_ref = ax.contour(X1, X2, X3, levels=[reference_level], colors=["r"])
        ax.clabel(cs_ref, inline=1)

        if save:
            plt.savefig(save_title)
        if show:
            plt.show()
        if close:
            plt.close()
        else:
            return fig, ax