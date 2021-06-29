import pandas as pd
import random
import matplotlib.pyplot as plt

def plot_elbow(list):
    centers = []
    dots = []
    # plot
    for element in list:
        centers.append(element[0])
        dots.append(element[1])

    plt.scatter(centers,dots)
    plt.show()


def plot_list(list):
    #data = list
    x = []
    y = []
    color = []
    colors = ("red", "green", "blue","yellow")

    for element in list:
        for k in range(len(element)):
            x.append(element[k])
            y.append(1)
            color.append(colors[(list.index(element) % len(colors))])


    for center in centers:
        x.append(center)
        y.append(1)
        color.append("black")

    print (x)
    print (y)
    print (color)

    plt.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30)

    plt.title('Matplot scatter plot')
    plt.legend(loc=2)
    plt.show()

'''
def plot_list(list):
    x = []
    y = []
    i=0
    for element in list:
        for k in range (len(element)):
            x.append(centers[i])
            y.append(element[k])
        i+=1

    print (x,y)
    plt.scatter(x,y,color='red')
    plt.show()
'''

def read_data_files(name):
    #return pd.read_csv("sample.csv")
    return pd.read_csv("Clustering.csv")

def update_centers(list):
    ans = 0.0
    new_centers = []
    i=0
    for element in list:
        ans=sum(element)/len(element)

        new_centers.append(ans)
  #  print (new_centers)
    return new_centers


def update_diff(list):
     ans = 0.0
     new_centers = []
     i = 0
     for element in list:
         for item in element:
            ans += ((item-centers[i])**2)
         i+=1

    #     print("this is the diff:", ans )
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

 #  print (total)
   return total

def pick_start_points(num_of_centers, x_array):
    return random.sample(list(x_array), num_of_centers)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("starting to read data files")
    clustering_dataset = read_data_files('self')
    for i in range(1, 10):
        d_in_prog =  clustering_dataset['Machine.num.'+str(i)]
        K=5
        converged =[]
        results = []

        for i in range(3,8):
          old_diff = 20000.0
          new_diff = 10000.0
          ans = []
          centers = pick_start_points(i, d_in_prog)
          while ans not in results:
              old_diff=new_diff
              if (ans):
                results.append(ans)
              ans = classify(d_in_prog)
             # print (ans)
              new_diff = update_diff(ans)
              centers = update_centers(ans)
          converged.append((i, new_diff))
        print (converged)
        plot_list(ans)
        plot_elbow(converged)

