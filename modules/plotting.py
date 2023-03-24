from typing import List, Tuple, Union, Optional, Dict, Any
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import silhouette_score, silhouette_samples
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def scatterplot_2D(name: str, labels_final: np.ndarray, embedding: np.ndarray) -> None:
    """
    Create a 2D scatter plot of embeddings with colors corresponding to cluster labels.

    Args:
        name (str): The name of the plot.
        labels_final (np.ndarray): A 1D array of cluster labels.
        embedding (np.ndarray): A 2D array of embeddings.

    Returns:
        None
    """
    # sns.set_context("talk", font_scale=1.1)
    # sns.set_style("whitegrid")
    plt.subplots(figsize = (8, 6))
    
    # g = sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], size="size", hue="size", alpha=0.5,
    #                     sizes=(100, 1000), palette="viridis", edgecolors="black")
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels_final.astype(int), s=0.1, cmap='Spectral')

    plt.tick_params(labelsize=12)

    # plt.legend(fancybox=True,framealpha=0.5,fontsize='15',loc='best', title_fontsize='30',
    #         bbox_to_anchor=(1., 1.))

    # plt.title(f"Clustering", fontsize=20)
    plt.xlabel("embedding_1", fontsize=15);plt.ylabel("embedding_2", fontsize=15)
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
        sil_bysample = silhouette_samples(embedding, labels_final, metric='cosine')
        sil_svg = round(float(silhouette_score(embedding, labels_final, metric='cosine')),3)
        
        y_lower = 10
        y_tick_pos_ = []

        fig, (ax) = plt.subplots(1)
        fig.set_size_inches(8, 12)

        for i in range(model.n_components):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_SILs = sil_bysample[labels_final == i]
            ith_cluster_SILs.sort()

            size_cluster_i = ith_cluster_SILs.shape[0]

            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / model.n_components)
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

        ax.set_title(f"Silhouette Plot for {model.n_components} clusters", fontsize=20)

        l_xlim = max(-1, min(-0.1, round(min(sil_bysample) - 0.1, 1)))
        u_xlim = min(1, round(max(sil_bysample) + 0.1, 1))
        ax.set_xlim([l_xlim, u_xlim])
        ax.set_ylim([0, embedding.shape[0] + (model.n_components + 1) * 10])
        ax.set_xlabel("Silhouette coefficient values", fontsize=20)
        ax.set_ylabel("Cluster label", fontsize=20)
        ax.set_yticks(y_tick_pos_)
        ax.set_yticklabels([str(i) for i in range(model.n_components)]) #,fontdict={'fontsize':15}
        ax.tick_params(axis='both', labelsize=15)   
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        # ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(f'results/{self.name}/SIL_bycluster.png')

        # Only if 2 dimensions, plot 2D scatterplot
        if model.n_components == 2:
            logging.info('Generating 2D scatterplot plot')
            scatterplot_2D(self.name, labels_final, embedding)
