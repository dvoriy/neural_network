from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


# we don't know the true label of the clustring so we need to use internal Validity Measures:
# https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation

######################## Silhouette Coefficient################################
# Advantages
# The score is bounded between -1 for incorrect clustering and +1
# for highly dense clustering. Scores around zero indicate overlapping clusters.
#
# The score is higher when clusters are dense and well separated, which relates
# to a standard concept of a cluster.

# Drawbacks
# The Silhouette Coefficient is generally higher for convex clusters than other
# concepts of clusters, such as density based clusters like those obtained through DBSCAN.


##################### Calinski-Harabasz Index ################################
# Advantages
# The score is higher when clusters are dense and well separated, which
# relates to a standard concept of a cluster.
#
# The score is fast to compute.

# Drawbacks
# The Calinski-Harabasz index is generally higher for convex clusters
# than other concepts of clusters, such as density based clusters like those obtained through DBSCAN.

######################  Davies-Bouldin Index #########################################
#  Advantages
# The computation of Davies-Bouldin is simpler than that of Silhouette scores.
#
# The index is computed only quantities and features inherent to the dataset.
#
# Drawbacks
# The Davies-Boulding index is generally higher for convex clusters
# than other concepts of clusters, such as density based clusters like those obtained from DBSCAN.

# The usage of centroid distance limits the distance metric to Euclidean space.

def my_K_Means(dataset):
    cs = []
    results = []

    print("Staring Kmeans")
    for i in range(1, K):
        # creation of K KMEANS model
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X_principal)
        cs.append(kmeans.inertia_)
        results.append(kmeans.cluster_centers_)

    best_num_of_clusters = 0
    for i in range(K):
        if cs[i] / cs[i + 1] < 1.2:
            best_num_of_clusters = i + 1
            break

    print("plotting Elbow Method Kmeans")
    plt.plot(range(1, K), cs)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('CS')
    plt.show()
    plt.scatter(X_principal['P1'], X_principal['P2'],
                c=KMeans(n_clusters=best_num_of_clusters).fit_predict(X_principal))
    plt.title('The K-Means clusters')
    plt.show()

    print("plotting silhouette scores Kmeans")
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
    plt.title('The Silhouette score per # clusters With K-Means')
    plt.show()

    print("plotting Calinski Harabasz Index Kmeans")
    Calinski_Harabasz_Index = []
    k = []
    for n_cluster in range(2, 8):
        Calinski_Harabasz_Index.append(
            calinski_harabasz_score(X_principal, KMeans(n_clusters=n_cluster).fit_predict(X_principal)))
        k.append(n_cluster)
        # Plotting a bar graph to compare the results

    plt.bar(k, Calinski_Harabasz_Index)
    plt.xlabel('Number of clusters', fontsize=10)
    plt.ylabel('Calinski Harabasz Index ', fontsize=10)
    plt.title('The Calinski Harabasz Index score per # clusters With K-Means')
    plt.show()

    print("plotting davies bouldin score Kmeans")
    davies_bouldin = []
    k = []
    for n_cluster in range(2, 8):
        davies_bouldin.append(
            davies_bouldin_score(X_principal, KMeans(n_clusters=n_cluster).fit_predict(X_principal)))
        k.append(n_cluster)
        # Plotting a bar graph to compare the results

    plt.bar(k, davies_bouldin)
    plt.xlabel('Number of clusters', fontsize=10)
    plt.ylabel('Davies bouldin score Index ', fontsize=10)
    plt.title('The davies bouldin score per # clusters With K-Means')
    plt.show()


def my_DBSCAN(dataset):
    K=7
    labels_arr = []
    n_clusters_arr = []
    # creating DBSCAN model
    for i in range(1, K):
        clustering = DBSCAN(eps=4.0 / i, min_samples=10, n_jobs=-1).fit(X_principal)
        core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
        core_samples_mask[clustering.core_sample_indices_] = True
        labels = clustering.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        labels_arr.append(labels)
        n_clusters_arr.append(n_clusters_)

        n_noise_ = list(labels).count(-1)

    plt.scatter(X_principal['P1'], X_principal['P2'],
                c=labels)
    plt.title('The DBSCAN Clusters')
    plt.show()
    n_clusters_arr = n_clusters_arr[1:-2]
    labels_arr = labels_arr[1:-2]
    print(n_clusters_arr)


    print("plotting silhouette scores With DBDCAN")
    silhouette_scores = []
    k = []
    for n_cluster in n_clusters_arr:
        silhouette_scores.append(
            silhouette_score(X_principal, labels_arr[n_clusters_arr.index(n_cluster)]))
        k.append(n_cluster)
        # Plotting a bar graph to compare the results

    plt.bar(k, silhouette_scores)
    plt.xlabel('Number of clusters', fontsize=10)
    plt.ylabel('Silhouette Score', fontsize=10)
    plt.title('The Silhouette score per # clusters With DBSCAN')
    plt.show()

    print("plotting Calinski Harabasz Index DBSCAN")
    Calinski_Harabasz_Index = []
    k = []
    for n_cluster in n_clusters_arr:
        Calinski_Harabasz_Index.append(
            calinski_harabasz_score(X_principal, labels_arr[n_clusters_arr.index(n_cluster)]))
        k.append(n_cluster)
        # Plotting a bar graph to compare the results

    plt.bar(k, Calinski_Harabasz_Index)
    plt.xlabel('Number of clusters', fontsize=10)
    plt.ylabel('Calinski Harabasz Index ', fontsize=10)
    plt.title('The Calinski Harabasz Index score per # clusters With DBSCAN')
    plt.show()

    print("plotting davies bouldin score DBSCAN")
    davies_bouldin = []
    k = []
    for n_cluster in n_clusters_arr:
        davies_bouldin.append(
            davies_bouldin_score(X_principal, labels_arr[n_clusters_arr.index(n_cluster)]))
        k.append(n_cluster)
        # Plotting a bar graph to compare the results

    plt.bar(k, davies_bouldin)
    plt.xlabel('Number of clusters', fontsize=10)
    plt.ylabel('Davies bouldin score Index ', fontsize=10)
    plt.title('The davies bouldin score per # clusters With DBSCAN')
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

    my_DBSCAN(X_principal)
    my_K_Means(X_principal)


