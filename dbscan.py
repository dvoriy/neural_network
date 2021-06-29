#Final Project TAU neural networks.
#By Roy & Yossi D
import pandas as pd
import matplotlib.pyplot as plt

#flatten = lambda x: [i for row in x for i in row]

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
    epsilon = 1.70
    min_pts = 300.0
    dataframe = read_data_files('self')
    for i in range (1,10):
        clusters = []
        visited = []
        dataset = dataframe['Machine.num.'+str(i)]
        recursive_dbscan(dataset)
        plot_list(clusters)

