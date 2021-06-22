import pandas as pd
import random
import matplotlib.pyplot as plt

def plot_list(list):
    centers = []
    dots = []

    # plot
    for element in list:
        centers.append(element[0])
        dots.append(element[1])
    result = dots + centers

    plt.scatter(centers,dots,color='red')
    plt.show()


def read_data_files(name):
    return pd.read_csv("sample.csv")
  # return pd.read_csv("Clustering.csv")

def update_centers(list):
    ans = 0.0
    new_centers = []
    i=0
    for element in list:
        ans=sum(element)/len(element)

        new_centers.append(ans)
    print (new_centers)
    return new_centers


def update_diff(list):
     ans = 0.0
     new_centers = []
     i = 0
     for element in list:
         ans += ((sum(element)-(centers[i]*len(element)))**2)/len(element)
         i+=1

         print("this is the diff:", ans )
         new_centers.append(ans)
     return (sum(new_centers)/len(new_centers))


def classify(dots_list):
   ans = []
   res = []
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
   #print (ans)
   for center in centers:
       res = []
       for item in ans:
            if (center == item[0]):
                res.append(item[1])
                #print (res)
       total.append(res)

   #print (total)
   return total

def pick_start_points(num_of_centers, x_array):
    return random.sample(list(x_array), num_of_centers)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("starting to read data files")
    clustering_dataset = read_data_files('self')
    d_in_prog =  clustering_dataset['Machine.num.1']
    K=5

    old_diff = 20000.0
    new_diff = 10000.0
    centers = pick_start_points(K, d_in_prog)
    while old_diff - new_diff > 1 :
        old_diff=new_diff
        ans = classify(d_in_prog)
        new_diff = update_diff(ans)
        print (new_diff)
        centers = update_centers(ans)

    plot_list(ans)

