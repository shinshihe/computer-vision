from cv2 import cv2
import numpy as np
import os



def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print ("---  new folder created  ---")
 
	else:
		print ("---  There is a folder!  ---")
		

def file_names(path):
    list_a = []
    names = os.listdir(path)
    # count = 0
    for i in names:
        tmp = i.split('_')
        if len(tmp) >= 3:
            if i.split('_')[2] == 'rgb.png':
                list_a.append(i)
                # count = count + 1
    return list_a
    # print(list_ara,count)





def operator(img,name,fold_name,mode):
    ## convert to hsv
    list_p = []
    # if mode == (1 and 2):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # else:
    #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # cv2.imwrite('gg.png',hsv)
    #(132,112,56) (124,107,58) (152,106,58) (181,92,61) (116,101,74) (149,106,53) (175,83,60)  （59,91,63） （224,76,67） （56,100,57）
    #(29,193,52) (128,201,36)
    #mask = cv2.inRange(hsv,(40,70,20),(80,255,255))
    if mode == 1:
        mask = cv2.inRange(hsv,(30,70,80),(80,255,255))
    if mode == 2:
        # mask = cv2.inRange(hsv,(20,100,60),(80,255,180))
        mask = cv2.inRange(hsv,(20,70,55),(80,255,255))
    if mode == 3:
        mask = cv2.inRange(hsv,(25,60,80),(80,255,255))
        # mask = cv2.inRange(hsv,(100,0,145),(255,140,255))
    # mask = cv2.inRange(hsv,(40,60,80),(80,255,255))
    
    ## slice the green
    imask = mask>0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]
    
    gray = cv2.cvtColor(green,cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    tmp,gray = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
    if mode == (1 and 2):     
        gray = cv2.medianBlur(gray,3)
        gray = cv2.medianBlur(gray,5)
    else:
        gray = cv2.medianBlur(gray,5)
        gray = cv2.blur(gray,(5,5))

    contours, hier = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # print(contours)
    ori = np.copy(img)
    count = 0
    area_list = []

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        if (w < 4*h and h< 4*w):
            area_list.append(w*h)
    if len(area_list) > 30:
            area_list.sort(reverse = True)
            area_list = area_list[0:29]       
            
    mean_area =int( np.mean(area_list) -1)
    if mode == 1:
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            if w*h > mean_area/2 :
                if (w < 3*h and h < 3*w):
                    cv2.rectangle(ori, pt1=(x, y), pt2=(x + w, y + h),color=(255, 255, 255), thickness=3)
                    count = count + 1
    if mode == 2:
        b_list = []
        b,_,_ = cv2.split(img)
        blue_mean = np.mean(b)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            if w*h > mean_area/25:
                if (w < 4*h and h < 4*w):
                    tmp_img = img[y:y+h,x:x+w]
                    b,_,_ = cv2.split(tmp_img)
                    b_list.append(np.mean(b))
        standard = 2*np.mean(b_list) - min(b_list) 
        # print(standard,blue_mean)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            if w*h > mean_area/25:
                if (w < 4*h and h < 4*w):
                    tmp_img = img[y:y+h,x:x+w]
                    b,_,_ = cv2.split(tmp_img)
                    # if np.mean(b) < blue_mean:
                    if np.mean(b) < (blue_mean + standard)/2 :
                        
                        cv2.rectangle(ori, pt1=(x, y), pt2=(x + w, y + h),color=(255, 255, 255), thickness=3)
                        count = count + 1
    if mode == 3:
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            if w*h > 200:
                if (w < 4*h and h < 4*w):
                    cv2.rectangle(ori, pt1=(x, y), pt2=(x + w, y + h),color=(255, 255, 255), thickness=3)
                    count = count  + 1
    
    

    ## save 
    tmp = name.split('_')[1]
    cv2.imwrite(fold_name + '\\' + tmp + '_rect.png', ori)
    cv2.imwrite(fold_name + '\\' + tmp + '_green.png', green)
    cv2.imwrite(fold_name + '\\' + tmp + '_gray.png',gray)
    return count 
    # return percentage

      

## Read


# path1 = '\Plant\Ara2013-Canon\\'
path1 = '\Tray\Ara2012\\'
# path2 = '\Plant\Tobacco\\'
path2 = '\Tray\Ara2013-Canon\\'
path3 = '\Tray\Ara2013-RPi\\'
work_path = os.getcwd()#获得当前工作目录
fold_name1 = 'Ara12_Results'
fold_name2 = 'Ara13_Results'
fold_name3 = 'AraRPI_Results'
folder1 = work_path + '\Ara12_Results'
folder2 = work_path + '\Ara13_Results'
folder3 = work_path + '\AraRPI_Results'
mkdir(folder1)
mkdir(folder2)
mkdir(folder3)  
#cv2.imshow('gg',img)
# print(work_path + path1)
list_Ara12 = file_names(work_path + path1)
list_Ara13 = file_names(work_path + path2)
list_RPI = file_names(work_path + path3)
Area_Ara12 = []
Area_Ara13 = []
Area_RPI = []

for i in list_Ara12:
    img = cv2.imread(work_path+path1+i)
    result = operator(img,i,fold_name1,1)
    Area_Ara12.append(result)


for i in list_Ara13:
    img = cv2.imread(work_path+path2+i)
    result = operator(img,i,fold_name2,2)
    Area_Ara13.append(result)


for i in list_RPI:
    img = cv2.imread(work_path+path3+i)
    result = operator(img,i,fold_name3,3)
    Area_RPI.append(result)


print(Area_Ara12)
print(Area_Ara13)
print(Area_RPI)
# cv2.imwrite('Ara_Results\\'+list_Ara[0],img)
# cv2.imshow('gg',img)
# print(len(Area_Ara),len(Area_Toba))
# print(max(Area_Ara),min(Area_Ara))
# print(max(Area_Toba),min(Area_Toba))
# tmp = max(Area_Ara)
# count = 0
# for i in Area_Toba:
#     if i > tmp:
#         count = count + 1
# mistakes = []
# haha =[]
# for i in range(0,len(Area_Toba)):
#     if Area_Toba[i] < tmp:
#         mistakes.append(i)
#         haha.append(Area_Toba[i])

# # print(Area_Toba)
# print(count)
# print(mistakes)
# print(haha)

# cv2.waitKey(0)