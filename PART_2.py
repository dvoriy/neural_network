from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def my_K_Means(dataset):
    cs = []
    results = []

    for i in range(1, K):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X_principal)
        cs.append(kmeans.inertia_)
        results.append(kmeans.cluster_centers_)

    best_num_of_clusters = 0
    for i in range(K):
        if cs[i] / cs[i + 1] < 1.2:
            best_num_of_clusters = i + 1
            break

    plt.plot(range(1, K), cs)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('CS')
    plt.show()
    plt.scatter(X_principal['P1'], X_principal['P2'],
                c=KMeans(n_clusters=best_num_of_clusters).fit_predict(X_principal))
    plt.title('The K-Means clusters')
    plt.show()

    silhouette_scores = []
    k = []
    for n_cluster in range(2, 8):
        silhouette_scores.append(
            silhouette_score(X_principal, KMeans(n_clusters=n_cluster).fit_predict(X_principal)))
        k.append(n_cluster)
        # Plotting a bar graph to compare the results

    plt.bar(k, silhouette_scores)
    plt.xlabel('Number of clusters', fontsize=10)
    plt.ylabel('Silhouette Score', fontsize=10)
    plt.title('The Silhouette score per # clusters')
    plt.show()


def my_DBSCAN(dataset):
    K=7
    for i in range(1, K):
        clustering = DBSCAN(eps=4.0 / i, min_samples=10, n_jobs=-1).fit(X_principal)
        core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
        core_samples_mask[clustering.core_sample_indices_] = True
        labels = clustering.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

    plt.scatter(X_principal['P1'], X_principal['P2'],
                c=labels)
    plt.title('The DBSCAN Clusters')
    plt.show()


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"]="10"
    print("starting to read data files")
    clustering_dataset = pd.read_csv("Clustering.csv")
    K = 17


    pca = PCA(n_components=2)
    X_principal = pca.fit_transform(clustering_dataset)
    X_principal = pd.DataFrame(X_principal)
    X_principal.columns = ['P1', 'P2']

    my_K_Means(X_principal)
    my_DBSCAN(X_principal)

