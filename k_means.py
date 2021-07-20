import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def plot_elbow(list):
    centers = []
    dots = []
    # plot
    for element in list:
        centers.append(element[0])
        dots.append(element[1])

    plt.scatter(centers,dots)
    plt.title('K-Means convergence plot')
    plt.show()


def plot_list(list, mid_points):
    x = []
    y = []
    color = []
    colors = ("red", "green", "blue", "yellow")

    for element in list:
        cur_color = colors[(list.index(element) % len(colors))]
        for k in range(len(element)):
            x.append(element[k])
            y.append(1)
            color.append(cur_color)

    for center in mid_points:
        x.append(center)
        y.append(1)
        color.append("black")


    plt.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30)

    plt.title('K-Means Scatter plot')
    plt.show()


def read_data_files(name):
    return pd.read_csv("Clustering.csv")

def update_centers(list):
    new_centers = []
    for element in list:
        ans=sum(element)/len(element)
        new_centers.append(ans)
    return new_centers


def update_diff(list,centers):
     ans = 0.0
     new_centers = []
     i = 0
     for element in list:
         for item in element:
            ans += ((item-centers[i])**2)
         i+=1

         new_centers.append(ans)
     return (sum(new_centers)/len(new_centers))


def classify(dots_list):
   ans = []
   total = []
   for dot in dots_list:
       min_dist = 1000000
       current_center = 0
       for center in centers:
            current_dist=abs(dot-center)
            if current_dist < min_dist:
                min_dist = current_dist
                current_center = center
       ans.append((current_center,dot))
   for center in centers:
       res = []
       for item in ans:
            if (center == item[0]):
                res.append(item[1])
       total.append(res)

   return total

def pick_start_points(num_of_centers, x_array):
    return random.sample(list(x_array), num_of_centers)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("starting to read data files")
    clustering_dataset = read_data_files('self')

    kmeans = KMeans(n_clusters=3, random_state=0)

    kmeans.fit(clustering_dataset)
    print(kmeans.cluster_centers_)
    for i in range(1, 10):
        d_in_prog = clustering_dataset['Machine.num.'+str(i)]
        K=17
        converged =[]
        results = []
        clusters = []
        mid_points = []

        for i in range(1,K):
          old_diff = 20000.0
          new_diff = 10000.0
          ans = []
          centers = pick_start_points(i, d_in_prog)
          while ans not in clusters:
              old_diff=new_diff
              if (ans):
                clusters.append(ans)
              ans = classify(d_in_prog)
              centers = update_centers(ans)
              new_diff = update_diff(ans,centers)
          converged.append((i, new_diff))
          results.append(ans)
          mid_points.append(centers)

        for i in range(K):
            if (converged[i][1]/converged[i+1][1] < 2):
                plot_list(results[i], mid_points[i])
                break
        plot_elbow(converged)

