#Final Project TAU neural networks.
#By Roy & Yossi D
import pandas as pd
import matplotlib.pyplot as plt
import sys
import operator
import time
from k_means import update_centers
from k_means import  update_diff

centers = []


def silhuette_width(list):
    a = []
    b = []
    S = []

    for cluster in list:
        for point in cluster:
            a.append(abs((sum(cluster) - len(cluster)*point)))
          #  print (a)
            min_dist_out_of_cluster=100000.0
            for neighbor_cluster in list:
                if neighbor_cluster == cluster:
                    continue
                for element in neighbor_cluster:
                    if abs(element - point) < min_dist_out_of_cluster:
                        min_dist_out_of_cluster = abs(element - point)

            b.append(min_dist_out_of_cluster)

    for val in map(operator.truediv, map(operator.sub, b, a), map(max, b, a)):
        S.append(val)

    if len(list) == 1:
        return 0
    if list:
        print (sum(S)/len(S))
        return (sum(S)/len(S))
    else:
        return -1

def read_data_files(name):
    #return pd.read_csv("sample.csv")
    return pd.read_csv("Clustering.csv")

def is_core(element, list):
    return (len(find_neighbors(element, list)) > min_pts)

def find_neighbors(element, list):
    neighbors = []
    for neighbor in list:
        if neighbor == element:
            continue
        if neighbor in visited:
            continue
        if abs(neighbor - element) < epsilon:
            neighbors.append(neighbor)
    return neighbors

def density_connected(element,k,list):
    neighbors = find_neighbors(element, list)
    for neighbor in neighbors:
        index.update({neighbor: k})
        if neighbor in core:
            if index[neighbor] == -1:
                density_connected(neighbor, k, list)





def scanner(list):
    noise = []
    for element in list:
        index.update({element : -1})
        if is_core(element , list):
            core.append(element)
    k = 0
    for element in core:
        if index[element] == -1:
            k = k+1
            index.update({element : k})
            density_connected(element,k,list)
  #  print(index)






def recursive_dbscan(list):
    for element in list:
        core = []
        neighbors = []
        if element in visited:
            continue
        visited.append(element)
        if is_core(element, list):
            core.append(element)
            neighbors.extend(find_neighbors(element,list))
            for neighbor in neighbors:
                visited.append(neighbor)
                core.append(neighbor)


            clusters.append(core)
            core = []

def align_clusters(index):
    cluster = []
    created_clusters = []
    for element in index:
        if index[element] in created_clusters:
            continue
        if index[element] == -1:
            continue
        cluster = []
        created_clusters.append(index[element])
        cluster.append(element)
        for neighbor in index:
            if element == neighbor:
                continue
            if index[neighbor] == index[element]:
                cluster.append(neighbor)
        clusters.append((cluster))
 #       print (clusters)
                #index.pop(neighbor)


def plot_list(list):
    #data = list
    x = []
    y = []
    color = []
    colors = ("red", "green", "blue","yellow")

    for element in dataset:
        x.append(element)
        y.append(0)
        color.append("orange")

    # Create plot
    for cluster in list:
        for element in cluster:
            x.append(element)
            y.append(1)
            color.append (colors[(list.index(cluster) % len(colors))])


    plt.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30)

    plt.title('DBSCAN scatter plot')
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    epsilon = 1
    min_pts = 50.0
    sys.setrecursionlimit(10 ** 6)
    s_w = -1.0


    dataframe = read_data_files('self')
    for i in range (1,10):
        to_continue = True
        while to_continue:
            clusters = []
            visited = []
            core = []
            index = {}
            dataset = dataframe['Machine.num.'+str(i)]

            ts = time.time()
            scanner(dataset)
            align_clusters(index)
            ts1 = time.time()
     #       centers = update_centers(clusters)
     #       print (centers)
     #       print (update_diff(clusters,centers))
            s_w_temp = silhuette_width(clusters)
            if s_w_temp < s_w:
                to_continue = False
            else:
                if s_w == s_w_temp:
                    break
                s_w = s_w_temp
                epsilon *= 2
                min_pts *= 2
            plot_list(clusters)


