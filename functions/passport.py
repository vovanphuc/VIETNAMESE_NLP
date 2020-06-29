import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


from tensorflow import keras
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

import pickle
def predict_DL(img_path):
    # 5. Định nghĩa model
    model = Sequential()
 
    # Thêm Convolutional layer với 32 kernel, kích thước kernel 3*3
    # dùng hàm sigmoid làm activation và chỉ rõ input_shape cho layer đầu tiên
    model.add(Conv2D(32, (3, 3), activation='sigmoid', input_shape=(28,28,1)))

    # Thêm Convolutional layer
    model.add(Conv2D(32, (3, 3), activation='sigmoid'))

    # Thêm Max pooling layer
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Flatten layer chuyển từ tensor sang vector
    model.add(Flatten())

    # Thêm Fully Connected layer với 128 nodes và dùng hàm sigmoid
    model.add(Dense(128, activation='sigmoid'))

    # Output layer với 10 node và dùng softmax function để chuyển sang xác xuất.
    model.add(Dense(10, activation='softmax'))
    # 6. Compile model, chỉ rõ hàm loss_function nào được sử dụng, phương thức 
    # đùng để tối ưu hàm loss function.
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    model = keras.models.load_model('model_ID/model_DL.h5')

    image = cv2.imread(img_path)
    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    preprocessed_digits = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        #print(x,y,w,h)
        if (w > 10 and h >10):
            cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
            digit = thresh[y:y+h, x:x+w]
            resized_digit = cv2.resize(digit, (18,18))
            padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
            preprocessed_digits.append(padded_digit)
    # plt.imshow(image, cmap="gray")
    # plt.show()    
    inp = np.array(preprocessed_digits)

    st =[]
    number = ''
    for digit in preprocessed_digits:
        prediction = model.predict(digit.reshape(1, 28, 28, 1))
        char = np.argmax(prediction)
        st.append(char)
    number = ' '.join([str(elem) for elem in st])
    print(number)
    return number
def predict_KNN(img_path):
    model = KNeighborsClassifier(n_neighbors=11)
    with open('model_ID/model_KNN.pkl', 'rb') as fid:
        model = pickle.load(fid)
    image = cv2.imread(img_path)
    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey.copy(), 90, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    preprocessed_digits = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # print(x,y,w,h)
        if (w > 10 and h > 10):
            cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            digit = thresh[y:y + h, x:x + w]
            resized_digit = cv2.resize(digit, (18, 18))
            padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

            # nbr = model.predict(padded_digit.reshape(1, 28, 28, 1))
            # cv2.putText(image, str(int(nbr[0])), (x, y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
            preprocessed_digits.append(padded_digit)
    #plt.imshow(image, cmap="gray")
    #plt.show()
    inp = np.array(preprocessed_digits)
    st = []
    number = ''
    for digit in preprocessed_digits:
        char = model.predict((digit.reshape(1, 784) / 255.0))[0]
        st.append(char)
    number = ' '.join([str(elem) for elem in st])
    print(number)
    return number
def predict_SVM(img_path):
    model = svm.SVC()
    with open('model_ID/model_SVM.pkl', 'rb') as fid:
        model = pickle.load(fid)
    image = cv2.imread(img_path)
    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey.copy(), 90, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    preprocessed_digits = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # print(x,y,w,h)
        if (w > 10 and h > 10):
            cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            digit = thresh[y:y + h, x:x + w]
            resized_digit = cv2.resize(digit, (18, 18))
            padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

            # nbr = model.predict(padded_digit.reshape(1, 28, 28, 1))
            # cv2.putText(image, str(int(nbr[0])), (x, y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
            preprocessed_digits.append(padded_digit)
    #plt.imshow(image, cmap="gray")
    #plt.show()
    inp = np.array(preprocessed_digits)
    st = []
    number = ''
    for digit in preprocessed_digits:
        char = model.predict((digit.reshape(1, 784) / 255.0))[0]
        st.append(char)
    number = ' '.join([str(elem) for elem in st])
    print(number)
    return number