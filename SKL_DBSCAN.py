from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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


    pca = PCA(n_components=2)
    X_principal = pca.fit_transform(clustering_dataset)
    X_principal = pd.DataFrame(X_principal)
    X_principal.columns = ['P1', 'P2']
    cs = []
    results = []
    for i in range(1, K):
        clustering = DBSCAN(eps=4.0/i, min_samples=10, n_jobs=-1).fit(X_principal)
        core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
        core_samples_mask[clustering.core_sample_indices_] = True
        labels = clustering.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print(n_clusters_)

    plt.scatter(X_principal['P1'], X_principal['P2'],
           c = DBSCAN(eps=4.0/5, min_samples=10, n_jobs=-1).fit(X_principal), cmap =plt.cm.winter)
    plt.show()