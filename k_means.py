import pandas as pd

def read_data_files(name):
    clustering_dataset = pd.read_csv("Clustering.csv")


class k_means_c:
    def __init__(self):
        self.data=clustering_dataset
        self.centers=pick_start_points()


    def pick_start_points(self, num_of_centers):
        list = random.sample(self.data, num_of_centers, *, counts=None)
        return list

    def find_cluster_centers(self):

    def find_clusters(self):
        pick_start_points(3)

class dbscan_c:
    def __init__(self):
        self.data=NULL


