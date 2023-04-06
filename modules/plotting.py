from typing import List, Tuple, Union, Optional, Dict, Any
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import silhouette_score, silhouette_samples
import seaborn as sns
from scipy.stats import multivariate_normal
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse

from sklearn.mixture import GaussianMixture

def plot_GMM(name, embedding, gmm, shadow=True):
    # determine number of clusters
    n_components = len(gmm.means_)
    labels = gmm.predict(embedding)

    # create colormap based on number of clusters
    cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, n_components)))

    # plot scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=3, cmap=cmap, alpha=0.5)

    # plot shadow ellipses
    if shadow:
        for i in range(n_components):
            mean = gmm.means_[i]
            cov = gmm.covariances_[i]
            if gmm.covariance_type == 'spherical':
                cov = np.diag(cov)
            elif gmm.covariance_type == 'diag':
                cov = np.diag(cov)
            elif gmm.covariance_type == 'tied':
                cov = gmm.covariances_
            elif gmm.covariance_type == 'full':
                cov = gmm.covariances_[i]
            else:
                raise ValueError("Invalid covariance type.")
            w, v = np.linalg.eigh(cov)
            angle = np.arctan2(v[0][1], v[0][0]) * 180 / np.pi
            width, height = 2 * np.sqrt(w)
            ell = Ellipse(mean, width, height, angle, alpha=0.3, facecolor=cmap(i))
            ax.add_artist(ell)

    ax.set_title('Gaussian Mixture Model Clusters')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    
    plt.savefig(f'results/{name}/scatterplot2D.png')

def scatterplot_2D(name: str, model: Any, embedding: np.ndarray) -> None:
    """
    Create a 2D scatter plot of embeddings with colors corresponding to cluster labels.

    Args:
        name (str): The name of the plot.
        model (Any): Fitted clustering model
        embedding (np.ndarray): A 2D array of embeddings.

    Returns:
        None
    """
    model_name = type(model).__name__
    labels = model.predict(embedding)
    n_clusters = len(set(labels))

    # sns.set_context("talk", font_scale=1.1)
    sns.set_style("white")

    fig, ax = plt.subplots(figsize=(8, 8))
    # create colormap based on number of clusters viridis/spectral
    cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, n_clusters)))
    # Create scatterplot
    ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=cmap, s=10, alpha=0.5)
      
    if model_name == 'GaussianMixture':
        for i in range(n_clusters):
            mean = model.means_[i]
            cov = model.covariances_[i]
            if model.covariance_type == 'spherical':
                cov = np.diag(cov)
            elif model.covariance_type == 'diag':
                cov = np.diag(cov)
            elif model.covariance_type == 'tied':
                cov = model.covariances_
            elif model.covariance_type == 'full':
                cov = model.covariances_[i]
            else:
                raise ValueError("Invalid covariance type.")
            w, v = np.linalg.eigh(cov)
            angle = np.arctan2(v[0][1], v[0][0]) * 180 / np.pi
            width, height = 2 * np.sqrt(w)
            ell = Ellipse(mean, width, height, angle, alpha=0.3, facecolor=cmap(i))
            ax.add_artist(ell)

    elif model_name == 'KMeans':
        centers = model.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], c="r", s=30, marker='x')
        
    plt.title(f"{model_name} clustering with K={n_clusters}", fontsize=20)
    plt.tick_params(labelsize=10)
    plt.xlabel("Component 1", fontsize=15);plt.ylabel("Component 2", fontsize=15)
    plt.autoscale()
    plt.tight_layout()
    plt.savefig(f'results/{name}/scatterplot2D.png')

class Plotting:
    def __init__(self, name: str):
        """
        Initialize the Plotting object.

        Args:
        - name (str): Name of the object.
        """
        self.name = name

    def elbow_plot_SIL(self, results_loop: pd.DataFrame, optimal_K: int) -> None:
        """
        Draw the elbow plot of SIL score vs. K.

        Args:
        - results_loop (pd.DataFrame): A pandas DataFrame with columns 'Clusters', 'Silhouette', and 'sil_stdv'.
        - optimal_K (int): The optimal number of clusters.

        Returns:
        None.
        """
        logging.info('Generating elbow plot')

        optimal_score = results_loop.loc[results_loop['Clusters'] == optimal_K, 'Silhouette'].values[0]
        optimal_stdv = results_loop.loc[results_loop['Clusters'] == optimal_K, 'sil_stdv'].values[0]
        optimal_label = f'Optimal K={optimal_K}\nSilhouette={optimal_score:.3f}±{optimal_stdv:.3f}'

        fig, ax1 = plt.subplots(figsize=(14, 6))

        sil = sns.lineplot(data=results_loop, x='Clusters', y="Silhouette", color='b', ci=None, estimator=np.median,
                        ax=ax1)

        sil_error = ax1.errorbar(x=results_loop['Clusters'], y=results_loop['Silhouette'], yerr=results_loop['sil_stdv'],
                                fmt='none', ecolor='b', capsize=4, elinewidth=1.5)

        plt.axvline(x=optimal_K, color='r', linestyle='--', label=optimal_label, linewidth=1.5)

        plt.legend(fancybox=True, framealpha=0.5, fontsize='15', loc='best', title_fontsize='30')
        plt.tick_params(labelsize=12)
        plt.title(f"Elbow plot - Sil vs. K", fontsize=20)
        plt.xlabel("Number of clusters (K)", fontsize=15)
        plt.ylabel("Silhouette", fontsize=15)
        plt.tight_layout()
        plt.savefig(f'results/{self.name}/{self.name}_Elbowplot.png')

    def distribution_plot(self, model: Any, embedding: np.ndarray) -> None:
        """
        Plot individual SIL scores each sample, agregated by cluster.

        Args:
        - model (Any): A clustering model (e.g., KMeans, DBSCAN).
        - embedding (np.ndarray): A numpy array of shape (n_samples, n_features) containing the data to be clustered.

        Returns:
        None.
        """
        
        logging.info('Generating distribution plot')

        labels_final = model.predict(embedding)
        n_clusters = len(set(labels_final))
        sil_bysample = silhouette_samples(embedding, labels_final, metric='euclidean')
        sil_svg = round(float(silhouette_score(embedding, labels_final, metric='euclidean')),3)
        
        y_lower = 10
        y_tick_pos_ = []

        fig, (ax) = plt.subplots(1)
        fig.set_size_inches(9, 12)

        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_SILs = sil_bysample[labels_final == i]
            ith_cluster_SILs.sort()

            size_cluster_i = ith_cluster_SILs.shape[0]

            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            # color = sns.color_palette("rocket", n_colors=model.n_components)#, as_cmap=True)

            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_SILs,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            y_tick_pos_.append(y_lower + 0.5 * size_cluster_i)

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        # ax.axvline(x=sil_svg, color="red", linestyle="--", label=f"Avg SIL score: {sil_svg}")
        ax.axvline(x=sil_svg, color="red", linestyle="--")

        ax.set_title(f"Silhouette Plot for {n_clusters} clusters", fontsize=20)

        l_xlim = max(-1, min(-0.1, round(min(sil_bysample) - 0.1, 1)))
        u_xlim = min(1, round(max(sil_bysample) + 0.1, 1))
        ax.set_xlim([l_xlim, u_xlim])
        ax.set_ylim([0, embedding.shape[0] + (n_clusters + 1) * 10])
        ax.set_xlabel("Silhouette coefficient values", fontsize=20)
        ax.set_ylabel("Cluster label", fontsize=20)
        ax.set_yticks(y_tick_pos_)
        ax.set_yticklabels([str(i) for i in range(n_clusters)]) #,fontdict={'fontsize':15}
        ax.tick_params(axis='both', labelsize=15)   
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        # ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(f'results/{self.name}/SIL_bycluster.png')

        # Only if 2 dimensions, plot 2D scatterplot
        # Currently ONLY work for euclidean spaces
        if (embedding.shape[1] == 2):
            logging.info('Generating 2D scatterplot plot')
            scatterplot_2D(self.name, model, embedding)

        # CODE TO GET CONTOUR PLOTS in GMM - NOT USED
        # Get the means and covariances of the GMM model
        # means = model.means_
        # covariances = model.covariances_

        # # Define the range of x and y values based on the limits of the embedding
        # xmin, xmax = np.min(embedding[:,0]), np.max(embedding[:,0])
        # ymin, ymax = np.min(embedding[:,1]), np.max(embedding[:,1])
        # x = np.linspace(xmin, xmax, 100)
        # y = np.linspace(ymin, ymax, 100)
        # X, Y = np.meshgrid(x, y)
        # pos = np.empty(X.shape + (2,))
        # pos[:, :, 0] = X
        # pos[:, :, 1] = Y

        # # Evaluate the PDF of the GMM model at each point
        # Z = np.zeros(X.shape)
        # for k in range(n_clusters):
        #     rv = multivariate_normal(means[k], covariances[k])
        #     Z += model.weights_[k] * rv.pdf(pos)

        # # Create the contour plot
        # contour = ax.contour(X, Y, Z)
        # ax.clabel(contour, inline=True, fontsize=8)
        # ax.set_xlim([xmin, xmax])
        # ax.set_ylim([ymin, ymax])