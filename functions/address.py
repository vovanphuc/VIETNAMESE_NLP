#LIB
import numpy as np
import json
import cv2
import os, random
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
from keras.layers import multiply, Dense, Permute, Lambda, RepeatVector
import itertools
import editdistance
# from lib.random_eraser import get_random_eraser
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from random import randint
from PIL import Image
import json
import os
from sklearn.model_selection import KFold
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Bidirectional, Dropout, MaxPooling2D
from keras.layers import Reshape, Lambda, BatchNormalization
from keras import applications
from keras.layers.recurrent import LSTM
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.optimizers import Adadelta, Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
# from loader import TextImageGenerator, MAX_LEN, CHAR_DICT, SIZE, VizCallback, ctc_lambda_func, attention_rnn
import numpy as np
import tensorflow as tf
from keras import backend as K
import argparse
import os
import tensorflow as tf
# from crnn import get_model
# from loader import SIZE, MAX_LEN, TextImageGenerator, beamsearch, decode_batch
from keras import backend as K
import glob                                                                 
import argparse
# import json
# import numpy as np
random.seed(2018)

letters = " !\"#&\\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"
MAX_LEN = 70
WIDTH, HEIGHT = 1280, 64
SIZE = WIDTH, HEIGHT
CHAR_DICT = len(letters) + 1

chars = letters
wordChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"
corpus = ' \n '.join(json.load(open('labels.json')).values())
word_beam_search_module = tf.load_op_library('/home/phuc/Desktop/CTCWordBeamSearch/cpp/proj/TFWordBeamSearch.so')
mat=tf.placeholder(tf.float32)
beamsearch_decoder = word_beam_search_module.word_beam_search(mat, 25, 'Words', 0.1, corpus, chars, wordChars)


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255):
    def eraser(input_img):
        img_h, img_w, _ = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser

def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))

def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[x] if x < len(letters) else "", labels)))


def beamsearch(sess, y_pred):
    y_pred = y_pred.transpose((1, 0, 2))
    results = sess.run(beamsearch_decoder, {mat:y_pred[2:]})
    blank=len(chars)
    results_text = []
    for res in results:
        s=''
        for label in res:
            if label==blank:
                break
            s+=chars[label] # map label to char
        results_text.append(s)
    return results_text

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def attention_rnn(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    timestep = int(inputs.shape[1])
    a = Permute((2, 1))(inputs)
    a = Dense(timestep, activation='softmax')(a)
    a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret

# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

class VizCallback(keras.callbacks.Callback):
    def __init__(self, sess, y_func, text_img_gen, text_size, num_display_words=3):
        self.y_func = y_func
        self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words
        self.text_size = text_size
        self.sess = sess

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        while num_left > 0:
            word_batch = next(self.text_img_gen.next_batch())[0]
            num_proc = min(word_batch['the_inputs'].shape[0], num_left)
            # predict
            inputs = word_batch['the_inputs'][0:num_proc]
            pred = self.y_func([inputs])[0]
            decoded_res = beamsearch(self.sess, pred)#decode_batch(pred)
            # label
            labels = word_batch['the_labels'][:num_proc].astype(np.int32)
            labels = [labels_to_text(label) for label in labels]
            
            for j in range(num_proc):
                edit_dist = editdistance.eval(decoded_res[j], labels[j])
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(labels[j])

            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print('\nOut of %d samples:  Mean edit distance: '
              '%.3f Mean normalized edit distance: %0.3f'
              % (num, mean_ed, mean_norm_ed))

    def on_epoch_end(self, epoch, logs={}):
        batch = next(self.text_img_gen.next_batch())[0]
        inputs = batch['the_inputs'][:self.num_display_words]
        labels = batch['the_labels'][:self.num_display_words].astype(np.int32)
        labels = [labels_to_text(label) for label in labels]
         
        pred = self.y_func([inputs])[0]
        pred_beamsearch_texts = beamsearch(self.sess, pred)
        #pred_texts = decode_batch(pred)
        for i in range(min(self.num_display_words, len(inputs))):
            print("label: {} - predict: {}".format(labels[i], pred_beamsearch_texts[i]))

        self.show_edit_distance(self.text_size)

class TextImageGenerator:
    def __init__(self, img_dirpath, labels_path, img_w, img_h,
                 batch_size, downsample_factor, idxs, training=True, max_text_len=9, n_eraser=5):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.idxs = idxs
        self.downsample_factor = downsample_factor
        self.img_dirpath = img_dirpath                  # image dir path
        self.labels= json.load(open(labels_path)) if labels_path != None else None
        self.img_dir = sorted(os.listdir(self.img_dirpath))     # images list
#         self.img_dir = [i for i in (os.path.join(self.img_dirpath, f) for f in os.listdir(img_dirpath)) if os.path.isfile(i)]
        random.shuffle(self.img_dir)

        if self.idxs is not None:
            self.img_dir = [self.img_dir[idx] for idx in self.idxs]

        self.n = len(self.img_dir)                      # number of images
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.imgs = np.ones((self.n, self.img_h, self.img_w, 3), dtype=np.float16)
        self.training = training
        self.n_eraser = n_eraser
        self.random_eraser = get_random_eraser(s_l=0.004, s_h=0.005, r_1=0.01, r_2=1/0.01, v_l=-128, v_h=128)
        self.texts = []
        image_datagen_args = {
		'shear_range': 0.1,
		'zoom_range': 0.01,
		'width_shift_range': 0.001,
		'height_shift_range': 0.1,
		'rotation_range': 1,
		'horizontal_flip': False,
		'vertical_flip': False
	}
        self.image_datagen = ImageDataGenerator(**image_datagen_args)

    def build_data(self):
        print(self.n, " Image Loading start... ", self.img_dirpath)
        for i, img_file in enumerate(self.img_dir):
            img = image.load_img(self.img_dirpath + img_file, target_size=SIZE[::-1], interpolation='bicubic')
            img = image.img_to_array(img)
            img = preprocess_input(img)
            self.imgs[i] = img
            if self.labels != None: 
                self.texts.append(self.labels[img_file][:MAX_LEN])
            else:
                #valid mode
                self.texts.append('')
        print("Image Loading finish...")

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]].astype(np.float32), self.texts[self.indexes[self.cur_index]]

    def next_batch(self):
        while True:
            X_data = np.zeros([self.batch_size, self.img_w, self.img_h, 3], dtype=np.float32)     # (bs, 128, 64, 1)
            Y_data = np.zeros([self.batch_size, self.max_text_len], dtype=np.float32)             # (bs, 9)
            input_length = np.ones((self.batch_size, 1), dtype=np.float32) * (self.img_w // self.downsample_factor - 2)  # (bs, 1)
            label_length = np.zeros((self.batch_size, 1), dtype=np.float32)           # (bs, 1)

            for i in range(self.batch_size):
                img, text = self.next_sample()

                if self.training:
                    params = self.image_datagen.get_random_transform(img.shape)
                    img = self.image_datagen.apply_transform(img, params)
                    if randint(0, 1) == 1:
                        for _ in range(self.n_eraser):
                            img = self.random_eraser(img)
                        img = elastic_transform(img, 10, 2, 0.1)

                img = img.transpose((1, 0, 2))
                # random eraser if training
                X_data[i] = img
                Y_data[i,:len(text)] = text_to_labels(text)
                label_length[i] = len(text)

            inputs = {
                'the_inputs': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length,  # (bs, 1)
                'label_length': label_length  # (bs, 1)
            }

            outputs = {'ctc': np.zeros([self.batch_size])}   # (bs, 1)
            yield (inputs, outputs)

def maxpooling(base_model):
    model = Sequential(name='vgg16')
    for layer in base_model.layers[:-1]:
        if 'pool' in layer.name:
            pooling_layer = MaxPooling2D(pool_size=(2, 2), name=layer.name)
            model.add(pooling_layer)
        else:
            model.add(layer)
    return model

def get_model(input_shape, training, finetune):
    inputs = Input(name='the_inputs', shape=input_shape, dtype='float32')
    base_model = applications.VGG16(weights='imagenet', include_top=False)
    base_model = maxpooling(base_model)
    inner = base_model(inputs)

    inner = Reshape(target_shape=(int(inner.shape[1]), -1), name='reshape')(inner)
    inner = Dense(512, activation='relu', kernel_initializer='he_normal', name='dense1')(inner) 
    inner = Dropout(0.25)(inner) 
    inner = attention_rnn(inner)

    lstm1 = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1', dropout=0.25, recurrent_dropout=0.25))(inner)
    lstm2 = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2', dropout=0.25, recurrent_dropout=0.25))(lstm1)
    
    y_pred = Dense(CHAR_DICT, activation='softmax', kernel_initializer='he_normal',name='dense2')(lstm2)
    
    labels = Input(name='the_labels', shape=[MAX_LEN], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    
    y_func = K.function([inputs], [y_pred])
    
    if training:
        Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out).summary()
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out), y_func
    else:
        return Model(inputs=[inputs], outputs=y_pred)

def train_kfold(idx, kfold, datapath, labelpath,  epochs, batch_size, lr, finetune, name):
    sess = tf.Session()
    K.set_session(sess)
    K.clear_session()
    model, y_func = get_model((*SIZE, 3), training=True, finetune=finetune)
    ada = Adam(lr=lr)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)

    ## load data
    train_idx, valid_idx = kfold[idx]
    train_generator = TextImageGenerator(datapath, labelpath, *SIZE, batch_size, 16, train_idx, True, MAX_LEN)
    train_generator.build_data()
    valid_generator  = TextImageGenerator(datapath, labelpath, *SIZE, batch_size, 16, valid_idx, False, MAX_LEN)
    valid_generator.build_data()

    ## callbacks
    weight_path = 'model/{}_{}.h5'.format(name, idx)
    ckp = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    vis = VizCallback(sess, y_func, valid_generator, len(valid_idx))
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')

    if finetune:
        print('load pretrain model')
        model.load_weights(weight_path)

    model.fit_generator(generator=train_generator.next_batch(),
                    steps_per_epoch=int(len(train_idx) / batch_size),
                    epochs=epochs,
                    callbacks=[ckp, vis, earlystop],
                    validation_data=valid_generator.next_batch(),
                    validation_steps=int(len(valid_idx) / batch_size))
    K.clear_session()
    
def train(datapath, labelpath, epochs, batch_size, lr, finetune=False, name='model'):
    nsplits = 5

    nfiles = np.arange(len(os.listdir(datapath)))

    kfold = list(KFold(nsplits, random_state=2018).split(nfiles))
    for idx in range(nsplits):
        train_kfold(idx, kfold, datapath, labelpath, epochs, batch_size, lr, finetune, name)

def loadmodel(weight_path):
    model = get_model((*SIZE, 3), training=False, finetune=0)
    model.load_weights(weight_path)
    return model

def predict(model, datapath, output, verbose=15):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # K.set_session(tf.Session(graph=model.output.graph))
        # init = K.tf.global_variables_initializer()
        # K.get_session().run(init)
        # sess = tf.Session()
        # sess = tf.compat.v1.Session()
        # K.clear_session()
        K.set_session(sess)
        # Declare this as global:
        global graph
        graph = tf.get_default_graph()
        # Then just before you call in your model, use this
        with graph.as_default():
            # call you models here
            batch_size = 3
            models = glob.glob('{}/best_*.h5'.format(model))
            test_generator  = TextImageGenerator(datapath, None, *SIZE, batch_size, 32, None, False, MAX_LEN)
            test_generator.build_data() # sinh ra text ='' for each image
        
            y_preds = []
            for weight_path in models:
            
                print('load {}'.format(weight_path))
                model = loadmodel(weight_path)
                X_test = test_generator.imgs.transpose((0, 2, 1, 3))
                y_pred = model.predict(X_test, batch_size=2)
                y_preds.append(y_pred)
                print(y_preds)
                # for printing        
                decoded_res = beamsearch(sess, y_pred[:verbose])
                if (len(y_preds)!=0):
                    y_preds = np.prod(y_preds, axis=0)**(1.0/len(y_preds))
                    y_texts = beamsearch(sess, y_preds)
                    submit = dict(zip(test_generator.img_dir, y_texts))
                    with open(output, 'w', encoding='utf-8') as jsonfile:
                        json.dump(submit, jsonfile, indent=2, ensure_ascii=False)
    
                for i in range(len(decoded_res)):
                    print('{}: {}'.format(test_generator.img_dir[i], decoded_res[i]))
                    return decoded_res[i]    

    K.clear_session()
    # if(len(y_preds)!=0):    
    #     y_preds = np.prod(y_preds, axis=0)**(1.0/len(y_preds))
    #     y_texts = beamsearch(sess, y_preds)
    #     submit = dict(zip(test_generator.img_dir, y_texts))
    #     with open(output, 'w', encoding='utf-8') as jsonfile:
    #         json.dump(submit, jsonfile, indent=2, ensure_ascii=False)

# if __name__=='__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", default='../data/ocr/model/', type=str)
#     parser.add_argument('--data', default='../data/ocr/preprocess/test/', type=str)
#     parser.add_argument('--output', default='../data/ocr/predict.json', type=str)
#     parser.add_argument('--device', default=2, type=int)
#     args = parser.parse_args()
    
    
#     os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)

#     predict(args.model, args.data, args.output)


# A= predict("model/", "image_test2/","predict.json")
# print('type:', type(A))
# print('predict:', A)
# print('version',tf.__version__)
def test(image):
    return image
