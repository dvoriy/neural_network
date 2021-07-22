from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


if __name__ == '__main__':
    OMP_NUM_THREADS = 10
    print("starting to read data files")
    clustering_dataset = pd.read_csv("Clustering.csv")
    K = 5


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

    plt.scatter(X_principal['P1'], X_principal['P2'],
           c = labels , cmap =plt.cm.winter)
    plt.show()




    silhouette_scores = []

    for n_cluster in range(2, K):
        silhouette_scores.append(silhouette_score(X_principal, DBSCAN(eps=4.0/n_cluster, min_samples=10, n_jobs=-1).fit_predict(X_principal)))


        # Plotting a bar graph to compare the results
    k = [3, 4, 5]
    plt.bar(k, silhouette_scores)
    plt.xlabel('Number of clusters', fontsize=10)
    plt.ylabel('Silhouette Score', fontsize=10)
    plt.show()