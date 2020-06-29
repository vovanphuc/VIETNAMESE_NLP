from flask import Flask, render_template, url_for, request, send_file, url_for
import re
import functions.address as address
import functions.passport as passport
import functions.cut_image as cut_image
import functions.processing_img as processing_img
from PIL import Image
import numpy as np
import json
import io
from tensorflow.keras.preprocessing.image import img_to_array
import cv2 
from werkzeug.utils import secure_filename
import functions.spell_correction as spell_correction
import functions.text_classifier as classifier
import os
path = '/home/phuc/Desktop/OCR_FLASK_BC/'
# STATIC_DIR = os.path.abspath('../static/styles')
app = Flask(__name__)

@app.route('/')
def index():
	return render_template("ocr.html")

@app.route('/id')
def id2():
	return render_template("id.html")

@app.route('/classifier')
def classify():
	return render_template("classifier.html")

@app.route('/spell')
def spell():
	return render_template("spell.html")

@app.route('/classifier', methods=['POST'])
def classify2():
	if request.method == 'POST':
		raw_text = request.form['rawtext']
		results = classifier.BigClassifier(raw_text)
	return render_template("classifier.html", results=results,raw_text=raw_text)

@app.route('/spell', methods=['POST'])
def spell2():
	if request.method == 'POST':
		raw_text = request.form['rawtext']
		results = spell_correction.correct_word(raw_text)
	return render_template("spell.html", results=results,raw_text=raw_text)

@app.route('/ocr', methods=['POST'])
def ocr_address():
	if request.method == "POST":
		image = request.files['image']
		name_img = secure_filename(image.filename)
		image = Image.open(image)
		image.save(path + 'image_test/address.png')
		#cut
		# cut_image.crop(path + 'image_test/address.png')
		# processing image
		# processing_img.processing(path + 'image_cut/cut.png')
		processing_img.processing(path + 'image_test/address.png')
		#address
		results_address1 = address.predict(path + "model_address_1/",path + "image_address/", path +"predict.json")
		results_address2 = address.predict(path + "model_address_2/",path + "image_address/", path +"predict.json")
		results_address3 = address.predict(path + "model_address_3/",path + "image_address/", path +"predict.json")
		# #cmnd
		# results_cmnd  = passport.predict(path + "image_dow/cmnd.png")
	# print('cmnd:', results_cmnd)
	return render_template("ocr.html", raw_text = name_img, address1= results_address1, address2= results_address2, address3 = results_address3)

@app.route('/id', methods=['POST'])
def id():
	if request.method == "POST":
		image = request.files['image']
		name_img = secure_filename(image.filename)
		image = Image.open(image)
		# cv2.imwrite('image_test2/form.jpg', image)
		image.save(path + 'image_test/cmnd.png')
		#PROCESSING
		processing_img.processing_id((path + 'image_test/cmnd.png'))
		# #cut
		# cut_image.crop(path + 'image_test2/form.png')
		# #address
		# results_address = address.predict(path + "model/",path + "image_address/", path +"predict.json")
		#cmnd
		results_cmnd1  = passport.predict_DL(path + "image_id/cmnd.png")
		results_cmnd2 = passport.predict_KNN(path + "image_id/cmnd.png")
		results_cmnd3 = passport.predict_SVM(path + "image_id/cmnd.png")
	return render_template("id.html", raw_text = name_img, ID1= results_cmnd1,ID2= results_cmnd2,ID3 = results_cmnd3 )
	
	# height = 64
	# width  = 1280
	# results = 0
	# # if request.files.get("image"):
	# if request.method == 'POST':
	# 	# get name picture
	# 	#get file image :user upload
	# 	image = request.files['image'].read()
	# 	#Convert image -> array image
	# 	image =  Image.open(io.BytesIO(image))
	# 	# image = image.resize((height,width))
	# 	image = img_to_array(image)
	# 	# image = np.expand_dims(image, axis=0)
	# 	print('image:', image.shape)
	# 	# cv2.imshow('1.jpg',image)
	# 	cv2.imwrite('../image_dow/'+ str(1)+'.png', image)
		
	
	# 	# # Predict
	# 	# # results = address.predict("model/",image, "predict.json")
	# 	# results = image.shape[1]
	# 	# print( 'shape:',image.shape)
	# else:
	# 	results = 120
	# return render_template("ocr.html", results = results, raw_text = 'ngu')
if __name__ == '__main__':
	 app.run(debug= False, threaded=False)	