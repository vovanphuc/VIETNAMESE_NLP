import cv2

def processing(img):
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    print(image.shape)
    blur = cv2.bilateralFilter(image, 5, 75, 75)
    img_binary = cv2.adaptiveThreshold(blur, 
                                       maxValue=255, 
                                       adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       thresholdType=cv2.THRESH_BINARY,
                                       blockSize=11,
                                       C=6)
    blur_next = cv2.GaussianBlur(img_binary,(3,3),0)
    imgScale = cv2.resize(blur_next, (int(1280), int(64)), interpolation = cv2.INTER_LINEAR)
    print(imgScale.shape)
    cv2.imwrite('image_address/address.png', imgScale)

def processing_id(img):
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        print(image.shape)
        blur = cv2.bilateralFilter(image, 5, 75, 75)
        img_binary = cv2.adaptiveThreshold(blur,
                                           maxValue=255,
                                           adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                           thresholdType=cv2.THRESH_BINARY,
                                           blockSize=11,
                                           C=2)
        blur_next = cv2.GaussianBlur(img_binary, (3, 3), 0)
        # imgScale = cv2.resize(blur_next, (int(1280), int(64)), interpolation=cv2.INTER_LINEAR)
        print(blur_next.shape)
        cv2.imwrite('image_id/cmnd.png', blur_next)