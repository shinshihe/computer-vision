# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 23:05:46 2020

@author: 
"""

#......IMPORT .........
import argparse
from cv2 import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

# find mean value
def find_value(image,threshold):
    biggersum = 0
    smallersum = 0
    biggersum = np.mean(np.ma.masked_greater_equal(image,threshold))
    smallersum = np.mean(np.ma.masked_less(image,threshold))
    value = (biggersum+smallersum)/2
    return value

# change image into binary image
def image_change(image,value):
    a,b = image.shape
    for i in range(0,a):
        for j in range(0,b):
            if image[i,j] >= value:
                image[i,j] = 0
            else:
                image[i,j] = 255
    return image

def find_threshold(image, threshold):
    thrlist = []
    while(1):
        thrlist.append(threshold)
        value = find_value(image,threshold)
        value = round(value)
        if (abs(threshold - value) >=1):
            threshold = value
        else:
            break
    return thrlist

# median filter, filter size can be changed by size
def median(image,size):
    a,b = image.shape
    f_size = (size-1)/2
    f_size = int(f_size)
    new_image = np.array(image,copy = True)
    for i in range(0,a-size):
        for j in range(0,b-size):
            value = []
            for m in range(i,i+size):
                for n in range(j,j+size):
                    value.append(image[m,n])
            value = sorted(value)
            index = (size*size + 1)/2
            index = int(index)
            new_image[i+f_size,j+f_size] = value[index]
    n = f_size
    for i in range(0,n):
        for j in range(0,b):
            new_image[i,j] = 255
    for i in range(a-n,a):
        for j in range(0,b):
            new_image[i,j] = 255
    for j in range(0,n):
        for i in range(0,a):
            new_image[i,j] = 255
    for j in range(b-n,b):
        for i in range(0,a):
            new_image[i,j] = 255
    # change the pixels on the edges to white
    # because the median filter doesn't work on these pixels
    return new_image

def check_neighbor(image,m,n):
    neighbor_pixels = []
    for i in range(n-1,n+2): #check the three pixels above image[m,n]
        if image[m - 1,i] > 1:
            neighbor_pixels.append(int(image[m - 1,i]))
    if image[m,n-1] > 1:
        neighbor_pixels.append(int(image[m,n - 1]))# check the two pixels beside image[m,n]
    if image[m,n+1] > 1:
        neighbor_pixels.append(int(image[m,n + 1]))
    if len(neighbor_pixels) >= 1:
        neighbor_pixels = sorted(neighbor_pixels)
        return neighbor_pixels[0],neighbor_pixels # find the minimum lable except 0
    return 0,neighbor_pixels

def neighbors(nlist,tmp):
    if len(tmp) > 1:
        tmp = list(set(tmp))
    if tmp not in nlist:
        nlist.append(tmp)
    return nlist
    
    
# step1 for two-pass
def two_pass(image,n):
    lable = 1
    n = int((n + 1)/2)
    #lable = int(lable)
    neighbor_list = []
    a,b = image.shape
    lable_image = np.zeros((a,b))
    for i in range(n,a-n):
        for j in range(n,b-n):
            if image[i,j] == 0:
                flag,neighbor_pixels = check_neighbor(lable_image,i,j)
                if flag == 0:
                    lable = lable + 1
                    lable_image[i,j] = lable
                    neighbor_pixels.append(lable)
                    neighbor_list = neighbors(neighbor_list,neighbor_pixels)
                else:
                    lable_image[i,j] = flag
                    neighbor_list = neighbors(neighbor_list,neighbor_pixels)
                    
    return lable_image,neighbor_list

# turn list[[2],[2,3],[3,4]] into list[[2,3,4]]
def combine(a):
    for i in range(0,len(a)):
        for j in range(0,len(a)):
            if i != j:
                for k in a[i]:
                    if k in a[j]:
                        a[i] = list(set(a[i]).union(set(a[j])))
                        del a[j]
                        return 1,a
    return 0,a

def twopass_steptwo(image,lista,n): # change all the connected lables into the same value
    a,b = image.shape
    for i in range(n,a-n):
        for j in range(n,b-n):
            if image[i,j] > 1:
                for k in lista:
                    if image[i,j] in k:
                        image[i,j] = k[0]
                        break
    return image


def lable_list(list):
    tmp = []
    for i in list:
        tmp.append(i[0])
    return tmp

def full_rice(lable_image,lables,average_size):
    lable_num = []
    damaged_rice = []
    #x_value,y_value = np.where(lable_image > 1)
    #average_size = int(1/2 * len(x_value)/rice_num)
    for i in lables:
        x_value,y_value = np.where(lable_image == i)
        lable_num.append(len(x_value))
    for i in range(0,len(lable_num)):
        if lable_num[i] < average_size:
            damaged_rice.append(lables[i])
    for i in damaged_rice:
        lables.remove(i)
    return lables

def full_rice_im(lables,image):
    a,b = image.shape
    new_image = np.zeros((a,b))
    for i in range(0,a):
        for j in range(0,b):
            new_image[i][j] = 255

    for i in lables:
        x_value,y_value = np.where(image == i)
        for j in range(0,len(x_value)):
            new_image[x_value[j],y_value[j]] = 0
    return new_image



def task1(image_a,name):
    threshold = 100
    thrlist = find_threshold(image_a,threshold)
    #print(thrlist)
    value = thrlist[-1]
    image = image_change(image_a,value) # change the image into binary image
    x = range(len(thrlist))
    plt.bar(x,thrlist)
    for j in range(len(thrlist)):
        plt.text(x[j] - 0.5,thrlist[j] + 5,thrlist[j])
    plt.xlabel('iterating time')
    plt.ylabel('threshold value')
    plt.title('threshold value')
    plt.savefig(args.OP_folder + '/' + name + '_threshold value.png')
    # plt.show()
    threshold_img = cv.imread(args.OP_folder+ '/' + name + '_threshold value.png')

    return image,value,threshold_img
    
    
    
def task2(image,n):
    image = median(image,n)
    #cv.imshow('happy',image)
    lable_image ,neighbor_list = two_pass(image,n) # step1 for two-pass
    flag = 1
    while(flag):
        flag,neighbor_list = combine(neighbor_list)
    rice_num = len(neighbor_list) 
    # neighbor_list contains all the labels, and put them together
    # if they are connected,such as:
    # neighbor_list[[2,3],[4,5],[6],[7,8,9]]
    
    # change all the lables which are connected to the same value
    lable_image = twopass_steptwo(lable_image,neighbor_list,n) 
    
    # lables contains the lables' value left in the image
    # lables = lable_list(lable_image)
    lables = lable_list(neighbor_list)
    return image,lable_image,rice_num,lables
    
    
    

def task3(lable_image,lables):
    full_rice_lables = full_rice(lable_image,lables,args.min_area)
    full_rice_image = full_rice_im(full_rice_lables,lable_image)

    return full_rice_image,len(full_rice_lables)
    
    
    
    

    
my_parser = argparse.ArgumentParser()
my_parser.add_argument('-o','--OP_folder', type=str,help='Output folder name', default = 'OUTPUT')
my_parser.add_argument('-m','--min_area', type=int,action='store', required = True, help='Minimum pixel area to be occupied, to be considered a whole rice kernel')
my_parser.add_argument('-f','--input_filename', type=str,action='store', required = True, help='Filename of image ')
# Execute parse_args()
args = my_parser.parse_args()

# -------------------my code below--------------------#

# find the image name
tmp = args.input_filename.split('/')
tmp = tmp[0]
tmp = tmp.split('.')
file_name = tmp[0]



# n is the filter's size, I choose 5 because it seems better than 3 or 7
n = 5
image_a = cv.imread(args.input_filename,0)

image,threshold,threshold_img = task1(image_a,file_name)
t_image = np.tile(image,(1,1)) # copy image

bin_image,lable_image,num,lables = task2(image,n) # num is the number of rice

full_rice_image,full_rice_num = task3(lable_image,lables)

# create the subfolder
path = args.OP_folder
if not os.path.exists(path): # if folder already exist, don't need to create a new one
    os.mkdir(path)



# task1
print('------threshold value = {} ------'.format(threshold))
cv.imshow('task1_bin_image,threshold value = %d'%(threshold),t_image) # t_image is a binary image which has not been filtered
cv.imshow('task1_threshold_image',threshold_img)

tmp_path = path + '/' + file_name + '_Task1.png'
cv.imwrite(tmp_path,t_image)

# task2
print('------number of rice = %d ------'%num)
cv.imshow('task2_filtered_bin_image',bin_image) 

tmp_path = path + '/' + file_name  + '_Task2.png'
cv.imwrite(tmp_path,bin_image)

# task3
print('------number of full rice = %d ------ '%full_rice_num)
print('------percentage of damaged rice = {}------'.format(round( (num-full_rice_num) /num,2)) )
cv.imshow('task3_full_rice_image',full_rice_image)

tmp_path = path + '/' + file_name  + '_Task3.png'
cv.imwrite(tmp_path,full_rice_image)

# plt.subplot(1,2,1),plt.imshow(t_image),plt.title('binary image')
# plt.subplot(1,2,2),plt.imshow(threshold_img),plt.title('threshold image')
# plt.show()
cv.waitKey(0)
