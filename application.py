from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
#from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from pathlib import Path
import numpy as np
import cv2 as cv
import sklearn

application = Flask(__name__)
application.url_map.strict_slashes = False

dic = {0 : 'Omelette', 1 : 'Onion Rings', 2 :"Samosa", 3:'Soup', 4:'Spring Rolls'}

model=load_model("model/inceptionv3_model.h5")


def predict_label(img_path):
	img = cv.imread(img_path,0)
	img = cv.resize(img, (256,256))
	gabor_1 = cv.getGaborKernel((18, 18), 1.5, np.pi/4, 5.0, 1.5, 0, ktype=cv.CV_32F) 
	filtered_img_1 = cv.filter2D(img, cv.CV_8UC3, gabor_1)
	img2 = cv.merge((filtered_img_1,filtered_img_1,filtered_img_1))
	img = np.expand_dims(img2, axis=0)
	p = model.predict(img)
	x = np.argmax(p)
	return dic[x]


@application.route('/')
def index():
	return render_template('home.html')


@application.route("/submit", methods = ['GET', 'POST'])
def get_prediction():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)



	return render_template("home.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	application.run(debug = True)