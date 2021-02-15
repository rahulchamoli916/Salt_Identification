import os
import random
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms as T

# Flask utils
from flask import Flask, flash, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# trained model weights
model_path = 'unet_model.pth'

# Uploaded image folder path
file_uploader = 'static/uploaded'
# predicted image folder path
file_predictor = 'static/predicted'


# Load pre-trained model
from model import UnetModel

model1 = UnetModel()
model1.load_state_dict(torch.load(model_path))
model1.eval()
print('Model Loaded Succesfully')

def random_name_generator(filename):
  keep_ext = filename.split('.')[-1]
  random_code = random.randint(10000,99999)
  new_file_name = str(random_code) + "." + keep_ext
  new_file_name = secure_filename(new_file_name)
  return new_file_name


def model_prediction(image_path, trained_model):
    ''' This function predicts mask image'''
    chk_img = Image.open(image_path)
    transforms = T.Compose([T.Grayscale(), T.ToTensor()])
    image = transforms(chk_img)
    output = trained_model(image.unsqueeze(0)).squeeze()
    pred = torch.where(output<0., torch.zeros_like(output), torch.ones_like(output))
    return pred.squeeze()


# Define a flask 
app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file was uploaded.')
            return redirect(request.url)
        # Get the file from post request
        image_file = request.files['image']

        if image_file.filename == '':
            flash('No file was uploaded.')
            return redirect(request.url)

        if image_file:
            # Save the file to ./uploads
            # uploaded image file path
            file_path = os.path.join(file_uploader,image_file.filename)
            # save uploaded image
            image_file.save(file_path)

            # Make prediction
            preds = model_prediction(file_path, model1)
            # generate random image file name for predicted image
            # print(image_file) #<FileStorage: '2bfa664017.png' ('image/png')>
            pred_filename = random_name_generator(image_file.filename)
            # predicted image absolute path
            pred_filepath = os.path.join(file_predictor,pred_filename)
            # save predicted image
            plt.imsave(pred_filepath, preds)


            return render_template('index.html', uploaded_image=image_file.filename, predicted_image=pred_filename)
    return render_template('index.html', uploaded_image=None, predicted_image=None)

if __name__ == '__main__':
    
    app.run()


# pip3 install torch==1.7.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html