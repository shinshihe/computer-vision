from cv2 import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img_toy = cv.imread('toy.png')
img_wheel = cv.imread('wheel.png')
img_wagon = cv.imread('wagon.jpg')

# if we choose create(500), we will get more than 100 key points.
# I think it's too much for small images especially for toy.png
# so I change it to 3000
n = 3000
surf_toy = cv.xfeatures2d.SURF_create(n)
surf_wheel = cv.xfeatures2d.SURF_create(n)
surf_wagon = cv.xfeatures2d.SURF_create(n)

# get key pionts
kp_toy, des_toy = surf_toy.detectAndCompute(img_toy,None)
kp_wheel, des_wheel = surf_wheel.detectAndCompute(img_wheel,None)
kp_wagon, des_wagon = surf_wagon.detectAndCompute(img_wagon,None)

#use model 4 so marks can be more obvious to see
key = 4
img_t = cv.drawKeypoints(img_toy,kp_toy,None,(),key)
img_wh = cv.drawKeypoints(img_wheel,kp_wheel,None,(),key)
img_wa = cv.drawKeypoints(img_wagon,kp_wagon,None,(),key)

# create BFMatcher

# bf = cv.BFMatcher(cv.NORM_L1,crossCheck = False)
# match_toy = bf.match(des_toy,des_wagon)
# match_wheel = bf.match(des_wheel,des_wagon)
# match_toy = sorted(match_toy, key = lambda x:x.distance)
# match_wheel = sorted(match_wheel, key = lambda x:x.distance)

# can't just simply use match, or all the points in wheel and toy
# will match with some points in wagon, even if these points are 
# really different
bf = cv.BFMatcher()
match_toy = bf.knnMatch(des_toy,des_wagon, k = 2)
match_wheel = bf.knnMatch(des_wheel,des_wagon, k = 2)

fine_match_toy = []
fine_match_wheel = []

# those matches who are too different will be excluded
for m,n in match_toy:
    if m.distance < 0.75* n.distance:
        fine_match_toy.append(m)

for m,n in match_wheel:
    if m.distance < 0.75* n.distance:
        fine_match_wheel.append(m)

# compare the ratio
if len(fine_match_wheel)/len(match_wheel) > len(fine_match_toy)/len(match_toy):
    better_img = img_wheel
    better_kp = kp_wheel
    better_match = fine_match_wheel
    print('wheel.png matches better')
else:
    better_img = img_toy
    better_kp = kp_toy
    better_match = fine_match_toy
    print('toy.png matches better')


match_img = cv.drawMatches(better_img,better_kp,img_wagon,kp_wagon,better_match,None,flags=2)

print('toy.png has %d keypoints'%len(kp_toy))
print('wheel.png has %d keypoints'%len(kp_wheel))
print('wagon.jpg has %d keypoints'%len(kp_wagon))

# store the images
cv.imwrite('toy_key_points.png',img_t)
cv.imwrite('wheel_key_points.png',img_wh)
cv.imwrite('wagon_key_points.png',img_wa)

#plot the images
plt.subplot(231),plt.imshow(img_t[...,::-1]),
plt.title('numbers of features = %d'%len(kp_toy)),plt.axis('off')

plt.subplot(232),plt.imshow(img_wh[...,::-1])
plt.title('numbers of features = %d'%len(kp_wheel)),plt.axis('off')

plt.subplot(233),plt.imshow(img_wa[...,::-1])
plt.title('numbers of features = %d'%len(kp_wagon)),plt.axis('off')

plt.subplot(234),plt.imshow(match_img[...,::-1])
plt.title('better matched image'),plt.axis('off')

plt.subplot(235),plt.imshow(better_img[...,::-1])
plt.title('original image'),plt.axis('off')

plt.subplot(236),plt.imshow(img_wagon[...,::-1])
plt.title('original image'),plt.axis('off')
plt.savefig('plot images')
plt.show()
cv.waitKey(0)

