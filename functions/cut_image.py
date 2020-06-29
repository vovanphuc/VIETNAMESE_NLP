import numpy as np
import cv2

# image = cv2.imread('/home/truc/Desktop/form.png')
# print(image.shape)
# def crop_name():
#   name = image[150:192, 207:905]
#   cv2.imshow('Name:',name)
#   cv2.imwrite('name.png',name)
# def crop_passport():
#   passport = image[229:272,207:905]
#   cv2.imshow('CMND:',passport)
#   cv2.imwrite('cmnd.png',passport)
# def crop_address():
#   address = image[314:357,207:905]
#   cv2.imshow('address:',address)
#   cv2.imwrite('address.png',address)
# Crop picture form
def crop(img):    
    image = cv2.imread(img)
    address = image[165:195,38:439]
    cv2.imwrite('image_cut/cut.png',address)
    # cv2.imwrite('image_address/address.png',address)
# cv2.waitKey(0)
# save image