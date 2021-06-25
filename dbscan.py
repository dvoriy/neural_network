#Final Project TAU neural networks.
#By Roy & Yossi D
import pandas as pd
import matplotlib.pyplot as plt

flatten = lambda x: [i for row in x for i in row]

def read_data_files(name):
    return pd.read_csv("Clustering.csv")

def is_core(element, list):
    return (len(find_neighbors(element, list)) > min_pts)


def find_neighbors(element, list):
    neighbors = []
    for neighbor in list:
        if neighbor in visited:
            continue
        if abs(neighbor - element) < epsilon:
            neighbors.append(neighbor)
    return neighbors

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


def plot_list(list):
    #data = list
    x = []
    y = []
    color = []
    colors = ("red", "green", "blue","yellow")
    groups = ("coffee", "tea", "water","blood")

    # Create plot
    for cluster in list:
        for element in cluster:
            x.append(element)
            y.append(1)
            color.append (colors[(list.index(cluster) % len(colors))])

    plt.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=groups)

    plt.title('DBSCAN scatter plot')
    plt.legend(loc=2)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    clusters = []
    epsilon = 1.0
    min_pts = 300.0
    visited = []
    dataframe = read_data_files('self')
    dataset = dataframe['Machine.num.3']

    recursive_dbscan(dataset)
    plot_list(clusters)
