import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

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
        The reference level 1 is plotted in red.

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

        # Plot the reference level set (1 thanks to the regularization)
        cs_ref = ax.contour(X1, X2, X3, levels=[1], colors=["r"])
        ax.clabel(cs_ref, inline=1)

        ax.set_title(f"Level sets of {self.model.__class__.__name__} with degree={self.model.d} and regularization={self.model.regularization} ({self.model.regularizer})")

        if save:
            plt.savefig(save_title)
        if show:
            plt.show()
        if close:
            plt.close()
        else:
            return fig, ax
        
    def boundary(self, x, *args, n_x1=100, n_x2=100, show=True, save=False, save_title="CF_boundary.png", close=True, **kwargs):
        """
        Plot the boundary decision of the model.
        In green, the points where the decision function is positive (considered as inliers),
        in red, the points where the decision function is negative (considered as outliers).

        Parameters
        ----------
        x: np.ndarray
            The data points to plot, shape (n_samples, n_features=2).
        n_x1: int
            Number of points along the first dimension for the grid.
        n_x2: int
            Number of points along the second dimension for the grid.
        show: bool
            Whether to show the plot. Defaults to True.
        save: bool
            Whether to save the plot to a file. Defaults to False.
        save_title: str
            Title for the saved plot file. Defaults to "CF_boundary.png".
        close: bool
            Whether to close the plot or return it after saving or showing. Defaults to True.
        """
        assert x.shape[1] == self.model.__dict__["p"]
        # Make figure
        fig, ax = plt.subplots()
        
        # Make a grid and predict the values
        x1_margin = (np.max(x[:, 0]) - np.min(x[:, 0])) / 5
        x2_margin = (np.max(x[:, 1]) - np.min(x[:, 1])) / 5
        ax.set_xlim([np.min(x[:, 0]) - x1_margin, np.max(x[:, 0]) + x1_margin])
        ax.set_ylim([np.min(x[:, 1]) - x2_margin, np.max(x[:, 1]) + x2_margin])
        X1 = np.linspace(np.min(x[:, 0] - x1_margin), np.max(x[:, 0] + x1_margin), n_x1)
        X2 = np.linspace(np.min(x[:, 1]) - x2_margin, np.max(x[:, 1] + x2_margin), n_x2)
        x1, x2 = np.meshgrid(X1, X2)
        x3 = np.c_[x1.ravel(), x2.ravel()]
        X3 = self.model.predict(x3).reshape(x1.shape)

        colors = ["red", "black", "green"]
        norm = Normalize(vmin=-1, vmax=1)  # Set the midpoint at zero
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
        ax.contourf(X1, X2, X3, cmap=cmap, norm=norm, alpha=0.7)

        # Scatter the data points
        ax.scatter(x[:, 0], x[:, 1], marker='x', s=20, alpha=0.3, color='blue')

        ax.set_title(f"Boundary decision of {self.model.__class__.__name__} with degree={self.model.d} and regularization={self.model.regularization} ({self.model.regularizer})")

        if save:
            plt.savefig(save_title)
        if show:
            plt.show()
        if close:
            plt.close()
        else:
            return fig, ax