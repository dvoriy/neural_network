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

    print ( "List length:" + str(len(list)))
    for cluster in list:
        print ("cluster length:" + str(len(cluster)))
        for point in cluster:
            a.append(abs((sum(cluster) - len(cluster)*point)/len(cluster)-1))

            min_dist_out_of_cluster=100000.0
            for neighbor_cluster in list:
                if neighbor_cluster == cluster:
                    continue
                for element in neighbor_cluster:
                    cur_min=(abs(element - point)/len(neighbor_cluster))
                    if  cur_min < min_dist_out_of_cluster:
                        min_dist_out_of_cluster = cur_min

            b.append(min_dist_out_of_cluster)

    print(a)
    print(b)
    for val in map(operator.truediv, map(operator.sub, b, a), map(max, b, a)):
        S.append(val)
    print (S)
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


def add_neighbors_to_core(element,core,list):
    visited.append(element)
    neighbors = find_neighbors(element, list)
    if len(neighbors) > min_pts:
        core.append(element)
        for neighbor in neighbors:
            if neighbor in visited:
                continue
            add_neighbors_to_core(neighbor, core, list)


def recursive_dbscan(list):

    for element in list:
        core = []
        if element in visited:
            continue
        add_neighbors_to_core(element, core, list)
        if (core):
            clusters.append(core)



def plot_list(list):
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
        cur_color = colors[(list.index(cluster) % len(colors))]
        for element in cluster:
            x.append(element)
            y.append(1)
            color.append (cur_color)


    plt.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30)

    plt.title('DBSCAN scatter plot')
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    sys.setrecursionlimit(10 ** 6)

    dataframe = read_data_files('self')
    for i in range (1,10):
        to_continue = True
        epsilon = 0.3
        min_pts = 128.0
        s_w = -1.0
        while to_continue:
            clusters = []
            visited = []
            core = []
            index = {}
            dataset = dataframe['Machine.num.'+str(i)]

            ts = time.time()
            recursive_dbscan (dataset)

            s_w_temp = silhuette_width(clusters)
            print("s_w_temp is " + str(s_w_temp))
            if s_w_temp < s_w:
                to_continue = False
            else:
                if s_w == s_w_temp:
                    if s_w > -1:
                        break
                s_w = s_w_temp
               # epsilon *= 2
                min_pts /= 2
                print ("s_w is "+ str(s_w))
                print ("index is " +str(i))
                print ("epsilon is " +str(epsilon))
                print ("min_pts is " +str(min_pts))

            plot_list(clusters)


