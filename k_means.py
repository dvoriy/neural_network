import pandas as pd
import random
import matplotlib.pyplot as plt

def plot_list(list):
    centers = []
    dots = []
    # height, weight and country data
    height = [167, 175, 170, 186, 190, 188, 158, 169, 183, 180]
    weight = [65, 70, 72, 80, 86, 94, 50, 58, 78, 85]
    country = ['A', 'A', 'B', 'B', 'B', 'B', 'A', 'A', 'B', 'A']

    # color map for each category
    colors = {'A': 'orange', 'B': 'blue'}
    color_ls = [colors[i] for i in country]

    # plot
    for element in list:
        centers.append(element[0])
        dots.append(element[1])
    #colors_ls =
    plt.scatter(dots,centers)
    plt.xlabel("Weight (Kg)")
    plt.ylabel("Height (cm)")
    plt.title("Height v/s Weight")
    plt.show()

ans = []
def read_data_files(name):
   clustering_dataset = pd.read_csv("Clustering.csv")
   #print(clustering_dataset.head(10))
   #print ("Read data files successfully")

   #print (clustering_dataset['Machine.num.1'])
   centers = pick_start_points(3, clustering_dataset['Machine.num.1'])

   for dot in clustering_dataset['Machine.num.1']:
       min_dist = 1000000
       current_center = 0
       for center in centers:
            if dot==center:
                continue
            current_dist=abs(dot-center)
            if current_dist < min_dist:
                min_dist = current_dist
                current_center = center
       ans.append((current_center,dot))

   #print (ans)
   plot_list(ans)



#class k_means_c:
 #   def __init__(self):
        #self.data = clustering_dataset
        #self.centers = pick_start_points()


def pick_start_points(num_of_centers, x_array):
    return random.sample(list(x_array), num_of_centers)


   # def find_cluster_centers(self):
   #     return

    #def find_clusters(self):
    #    pick_start_points(3)

#class dbscan_c:
#    def __init__(self):
     #   self.data=NULL


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("starting to read data files")
    read_data_files('self')
  #  clustering_dataset.head(10)
