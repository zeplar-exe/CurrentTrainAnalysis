# per event, cluster subjects by their activation patterns to determine if there's discrete clusters of patterns

# the greater idea: what if we did some loss-based thing by looking at all of the individual samples of an event (across subjects, within the same subject cluster) to tune the coalesced output, since the coalesce is far too close for comfort (SD is too low)
    # loss is based on matching all of the same-cluster samples to each other, and then penalizing if same-cluster samples are far away from each other
        # perhaps we do contrastive stuff; penalize different-cluster samples that are too close
# brother what the fuck are you talking about? that's just a regular clustering algorithm with extra steps
    # oh I get it, it's just using silhouette score in a loop

# !!! double check SD and variance across events in JASP for per-subject and across-subject/coalesce
# !! also need to eyeball clusters in all subjects; 
# !! ALSO we need to start using the eye control dataset, and an error detection dataset
# need to look at overlap across per-band clusters

# by the way again: what if we split out by cluster and then pass the raw growth values into a supervised decoder on an ms basis? 

# by the way... how do we, like, force a cluster to appear multiple time intra subject, OR multiple times intersubject to be considered a cluster?
# also: let's export raw (not pos or neg, raw), I want to test it for clustering (still using top percentile)
# also lets test clustering on https://www.kaggle.com/competitions/pnpl-competition-2026-deep/overview

import matplotlib.pyplot as plt
from coalesce_colonies import coalesce_colonies
from colony_viewer import show_colony
from loader import ColonyQuery, EventSet, SubjectSet, get_colonies
from core import BANDS, DATASET_SPECS, get_dataset_spec
from itertools import groupby
import numpy as np
import pandas as pd
from prince import MCA
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


EMBEDDING_COMPONENTS = 20
PERCENTILES = [75, 85, 95]

# https://scikit-learn.org/stable/auto_examples/cluster/plot_hdbscan.html
def plot(X, labels, probabilities=None, parameters=None, ground_truth=False, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    labels = labels if labels is not None else np.ones(X.shape[0])
    probabilities = probabilities if probabilities is not None else np.ones(X.shape[0])
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    # The probability of a point belonging to its labeled cluster determines
    # the size of its marker
    proba_map = {idx: probabilities[idx] for idx in range(len(labels))}
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_index = (labels == k).nonzero()[0]
        for ci in class_index:
            ax.plot(
                X[ci, 0],
                X[ci, 1],
                "x" if k == -1 else "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=4 if k == -1 else 1 + 5 * proba_map[ci],
            )
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    preamble = "True" if ground_truth else "Estimated"
    title = f"{preamble} number of clusters: {n_clusters_}"
    if parameters is not None:
        parameters_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f" | {parameters_str}"
    ax.set_title(title)
    plt.tight_layout()

for percentile in PERCENTILES:
    for event in get_dataset_spec("grasplift")["event_ids"].values():
        results = get_colonies(ColonyQuery(
            subjects=[SubjectSet.all("grasplift")],
            events=[EventSet("grasplift", [event])],
            bands=["theta", "delta", "alpha", "beta", "gamma"],
            sources=["inverse"],
            colony_types=["pos", "neg"]
        ))
        by_band = groupby(results, lambda c: (c.band, c.event))
        
        for (band, event), event_results in by_band:
            pos_binary_ownership = []
            neg_binary_ownership = []
            pos_colonies = []
            neg_colonies = []
            
            by_subject = groupby(event_results, lambda c: c.subject)
            
            for _, subject_results in by_subject:
                for result in subject_results:
                    if result.type == "pos":
                        o = result.colony.pos_weights()
                        o = o > np.percentile(o, percentile)
                        pos_binary_ownership.append(o)
                        pos_colonies.append(result.colony)
                    elif result.type == "neg":
                        o = result.colony.neg_weights()
                        o = o > np.percentile(o, percentile)
                        neg_binary_ownership.append(o)
                        neg_colonies.append(result.colony)

            mca = MCA(n_components=EMBEDDING_COMPONENTS, n_iter=3, copy=True, check_input=True, engine='sklearn', random_state=42, 
                      one_hot=False)
            df = pd.DataFrame(pos_binary_ownership)
            # dropping empty columns lest covariance matrix math fucks over and spits out NaNs
            df.drop(columns=df.columns[df.sum() == 0], inplace=True)
            print(df.head())
            mca.fit(df)
            embedding = mca.row_coordinates(df)
            print(embedding)
            
            hdb = HDBSCAN(min_cluster_size=2, min_samples=None, copy=True, store_centers="centroid")
            hdb.fit(embedding)
            
            if len(hdb.labels_) == 0 or (len(hdb.labels_) == 1 and -1 in hdb.labels_):
                print("No valid clusters found.")
                continue

            td_pca = TSNE(perplexity=10) # PCA(n_components=2)
            embedding_2d = td_pca.fit_transform(embedding)
            
            plot(
                embedding_2d,
                hdb.labels_,
                hdb.probabilities_,
                parameters={"band": band, "event": event, "percentile": percentile},
            )
            plt.show()
            clusters = groupby(zip(hdb.labels_, pos_colonies), lambda x: x[0])
            clusters = [coalesce_colonies(colonies) for i, colonies in clusters]
            show_colony(clusters[0], name=f"Band: {band}, Event: {event}, Percentile: {percentile}, Cluster 0")
        # + for top n% clustering: Multiple Correspondence Analysis (into HDBSCAN), K Modes Clustering, ROCK
        # for weight clustering: HDBSCAN (+ UMAP?)
        # + in either case, would help to do a 2D projection of clusters to visualize
        # and obviously, we need to coalesce the clusters themselves (presuming they're valid clusters) to see subject response clusters

