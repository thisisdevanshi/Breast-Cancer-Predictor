# virtualenv env
# .\env\Scripts\activate
#python app.py
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from time import time
import sys
import os
import re
import numpy as np

# Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.pipeline import Pipeline
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model 
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling3D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing import image
from glob import glob
import cv2
from sklearn.metrics import f1_score
import tensorflow_addons as tfa

# Import other libraries
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from pathlib import Path
import imageio as io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
from tensorflow.keras import backend as K

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Import Keras dependencies
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()
from tensorflow.keras.preprocessing import image

# Import other dependecies
import numpy as np
import h5py
from PIL import Image
import PIL
import os

#::: Flask App Engine :::
# Define a Flask app
app = Flask(__name__)

# ::: Prepare Keras Model :::
# Model files
binary_densenet = load_model('saved_models/binary_densenet')
multi_densenet = load_model('saved_models/multi_densenet')


print('Model loaded. Check http://127.0.0.1:5000/')


# ::: MODEL FUNCTIONS :::
def model_predict(img_path, binary_model, multi_model):
	'''
		Args:
			-- img_path : an URL path where a given image is stored.
			-- model : a given Keras CNN model.
	'''

	IMG = image.load_img(img_path, target_size=(512, 512))
	print(type(IMG))

	# Pre-processing the image
	# IMG_ = np.asarray(IMG)
	# print(IMG_.shape)
	IMG_ = image.img_to_array(IMG)
    # x = np.true_divide(x, 255)
	IMG_ = (IMG_-IMG_.mean())/(IMG_.std())
	IMG_ = np.expand_dims(IMG_, axis=0)
	print(type(IMG_), IMG_.shape)

	binary_prediction = binary_model.predict(IMG_)

	if binary_prediction>0.5:
		prediction_prop = np.append(binary_prediction.round(decimals=2),np.zeros(14))
	else:
		multi_pred = multi_model.predict(IMG_).round(decimals=2)
		prediction_prop = np.append(binary_prediction.round(decimals=2),multi_pred)

	return prediction_prop


# ::: FLASK ROUTES
@app.route('/', methods=['GET'])
def index():
	# Main Page
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():

	# Constants:
	classes = np.array(['No finding', 'Aortic enlargement',
       'Atelectasis', 'Calcification',
       'Cardiomegaly', 'Consolidation', 'ILD',
       'Infiltration', 'Lung Opacity',
       'Nodule/Mass', 'Other lesion',
       'Pleural effusion', 'Pleural thickening',
       'Pneumothorax', 'Pulmonary fibrosis'])

	if request.method == 'POST':

		# Get the file from post request
		f = request.files['file']

		# Save the file to ./uploads
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(
			basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)

		# Make a prediction
		prediction = model_predict(file_path, binary_densenet, multi_densenet)

		# If no_finding==1, then output normal
		if prediction[0]>0.5:
			predicted_class = 'normal - no findings'
		else:
			predicted_class = classes[prediction>0.5]
			if len(predicted_class)==0:
				predicted_class = 'normal - no findings'
				prediction[0]=0.51
		
		print('We think that is {}.'.format(predicted_class))

		return str(predicted_class)


if __name__ == '__main__':
	app.run(debug = True)




