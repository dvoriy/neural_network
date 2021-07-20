from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np

# https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    OMP_NUM_THREADS = 10
    print("starting to read data files")
    clustering_dataset = pd.read_csv("Clustering.csv")
    K = 5
    #kmeans = KMeans(n_clusters=3, random_state=0)

    cs = []
    results = []
    for i in range(1, K):
        clustering = DBSCAN(eps=4.0/i, min_samples=10, n_jobs=-1).fit(clustering_dataset)
        core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
        core_samples_mask[clustering.core_sample_indices_] = True
        labels = clustering.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print(n_clusters_)