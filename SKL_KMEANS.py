from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    OMP_NUM_THREADS = 10
    print("starting to read data files")
    clustering_dataset = pd.read_csv("Clustering.csv")
    K = 17
    #kmeans = KMeans(n_clusters=3, random_state=0)

    cs = []
    results = []
    for i in range(1, K):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(clustering_dataset)
        cs.append(kmeans.inertia_)
        results.append(kmeans.cluster_centers_)

    print(cs)
    for i in range(K):
        if cs[i] / cs[i + 1] < 1.2:
            print(results[i])
            break

    plt.plot(range(1, K), cs)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('CS')
    plt.show()

