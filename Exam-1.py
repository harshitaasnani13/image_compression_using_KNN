import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as pl
from skimage.util.shape import view_as_blocks
from image_slicer import slice
from sklearn.cluster import KMeans
from scipy.spatial import distance

#read image and convert it to array
image = Image.open('Carnegia-Interior.jpg')
array = np.asarray(image)

# divide image into arrays of 4*4*3 size
x=view_as_blocks(array,(4,4,3))
arr_ls=[]
for i in range(0,816):
    for j in range(0,612):
        arr_ls.append(x[i][j][0])


#k-means algorithm
def get_initial_mean(k):
    Z=[]
    for i in range(0,k):
        z=np.random.randint(1,200, size=(4,4,3))
        Z.append(z)
    return Z

def group(Z_new,element):
    dist=[]
    for i in range(0,len(Z_new)):
        dist.append(np.linalg.norm(Z_new[i]-element))
    ret = dist.index(min(dist))
    return ret


def get_group_array(Z_new,arraylist):
    grouplist=[]
    for i in range(0,len(arraylist)):
        grouplist.append(group(Z_new,arraylist[i]))
    return grouplist

def get_indices(k,grouplist):
    indices=[]
    for i in range (0, len(grouplist)):
        if grouplist[i]==k:
            indices.append(i)
    return indices

def get_array_elements(arraylist,indices):
    arrayelements = [arraylist[i] for i in indices]
    return arrayelements

def get_sum(arrayelements):
    s=0
    for i in range(0,len(arrayelements)):
        s=np.add(s,arr_ls[i],dtype=np.float)
    return s
def get_array_elements_mean(arrayelements):
    s = get_sum(arrayelements)
    l = len(arrayelements)
    mean = s
    if l!= 0:
        mean = s/l
    return mean

def calculate_new_centroid(arraylist,grouplist,k):
    Z_new=[]
    for i in range(0,k):
        indices= get_indices(i,grouplist)
        arrayelements = get_array_elements(arraylist,indices)
        mean=get_array_elements_mean(arrayelements)
        Z_new.append(mean)
    return Z_new

def k_means(arraylist,k,n_iter):
    Z_new = get_initial_mean(k)
    grouplist_old=[]
    grouplist_new=[]
    i=0
    for n in range (0, n_iter):
        i+=1
        grouplist_new = get_group_array(Z_new,arraylist)
        if n>0 and grouplist_new == grouplist_old:
            break
        grouplist_old=[]
        for cluster in grouplist_new:
            grouplist_old.append(cluster)
        Z_new = calculate_new_centroid(arraylist,grouplist_new,k)
    print(i)
    return (grouplist_new,Z_new)

z=k_means(arr_ls,200,5)

def replace_with_mean(z,arr_ls):
    for i in range(0,len(arr_ls)):
        group = z[0][i]
        rep_mean = z[1][group]
        arr_ls[i]=rep_mean
    return arr_ls
arr_ls=replace_with_mean(z,arr_ls)

#merge all arrays with replaced z vector to get image
y=np.copy(x)
for i in range(0,816):
    for j in range(0,612):
        y[i][j][0] = arr_ls[i*612 + j]
        
y=x.transpose(0,3,1,4,2,5).reshape(3264, 2448, 3)
pl.imshow(y)