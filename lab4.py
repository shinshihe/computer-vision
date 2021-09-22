

from sklearn import datasets
from cv2 import cv2

digits = datasets.load_digits()
print(digits.data.shape)
cv2.imshow('gg',digits.images[1])
cv2.waitKey(0)