#Final Project TAU neural networks.
#By Roy & Yossi D
import pandas as pd


def read_data_files(name):
    return pd.read_csv("sample.csv")
    #return pd.read_csv("Clustering.csv")

def classify(list):
    minpts=0
    #look for core points:
    for element in list:
        for neighbor in list:
            if abs(neighbor - element) < epsilon :
                minpts += 1
        if minpts > min_pts:
            core.append(element)
        minpts = 0

    #look for border points:
    for element in list:
        if element in core or element in border:
            continue
        for neighbor in list:
            if abs(neighbor - element) < epsilon and neighbor in core:
                border.append(element)


def is_core(element, list):
    minpts = 0
    for neighbor in list:
        if abs(neighbor - element) < epsilon:
            minpts += 1
    if minpts > min_pts:
        return True
    else:
        return False

def new_dbscan(list):
    for element in list:
        core = []
        if element in visited:
            continue
        visited.append(element)
        if is_core(element, list):
            core.append(element)

            for neighbor in list:
                if neighbor in visited:
                    continue
                if abs(neighbor - element) < epsilon:
                    visited.append(neighbor)
                    if is_core(neighbor, list):
                        core.append(neighbor)
            clusters.append(core)
            core = []


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    border = []
    clusters = []
    epsilon = 3.0040
    min_pts = 5.0
    visited = []
    dataframe = read_data_files('self')
    dataset = dataframe['Machine.num.1']
    new_dbscan(dataset)
    #classify(dataset)
    print (dataset)
    print (len(clusters))
    print (clusters)
