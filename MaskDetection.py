#Step 1

project_id = 'a459ed32-7bad-4d9e-9081-8d78a326aada' # Replace with your project ID
cv_key = 'c83e93f685f04808b22e542ddf9b8d62' # Replace with your prediction resource primary key
cv_endpoint = 'https://objectdata.cognitiveservices.azure.com/' # Replace with your prediction resource endpoint

model_name = 'Iteration3' # this must match the model name you set when publishing your model iteration exactly (including case)!
print('Ready to predict using model {} in project {}'.format(model_name, project_id))

#Ready to predict using model Iteration3 in project a459ed32-7bad-4d9e-9081-8d78a326aada

#Step 2
!pip install azure-cognitiveservices-vision-customvision

Requirement already satisfied: azure-cognitiveservices-vision-customvision in ./anaconda3/lib/python3.8/site-packages (3.1.0)
Requirement already satisfied: azure-common~=1.1 in ./anaconda3/lib/python3.8/site-packages (from azure-cognitiveservices-vision-customvision) (1.1.26)
Requirement already satisfied: msrest>=0.5.0 in ./anaconda3/lib/python3.8/site-packages (from azure-cognitiveservices-vision-customvision) (0.6.19)
Requirement already satisfied: certifi>=2017.4.17 in ./anaconda3/lib/python3.8/site-packages (from msrest>=0.5.0->azure-cognitiveservices-vision-customvision) (2020.6.20)
Requirement already satisfied: requests-oauthlib>=0.5.0 in ./anaconda3/lib/python3.8/site-packages (from msrest>=0.5.0->azure-cognitiveservices-vision-customvision) (1.3.0)
Requirement already satisfied: requests~=2.16 in ./anaconda3/lib/python3.8/site-packages (from msrest>=0.5.0->azure-cognitiveservices-vision-customvision) (2.24.0)
Requirement already satisfied: isodate>=0.6.0 in ./anaconda3/lib/python3.8/site-packages (from msrest>=0.5.0->azure-cognitiveservices-vision-customvision) (0.6.0)
Requirement already satisfied: oauthlib>=3.0.0 in ./anaconda3/lib/python3.8/site-packages (from requests-oauthlib>=0.5.0->msrest>=0.5.0->azure-cognitiveservices-vision-customvision) (3.1.0)
Requirement already satisfied: chardet<4,>=3.0.2 in ./anaconda3/lib/python3.8/site-packages (from requests~=2.16->msrest>=0.5.0->azure-cognitiveservices-vision-customvision) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in ./anaconda3/lib/python3.8/site-packages (from requests~=2.16->msrest>=0.5.0->azure-cognitiveservices-vision-customvision) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in ./anaconda3/lib/python3.8/site-packages (from requests~=2.16->msrest>=0.5.0->azure-cognitiveservices-vision-customvision) (2.10)
Requirement already satisfied: six in ./anaconda3/lib/python3.8/site-packages (from isodate>=0.6.0->msrest>=0.5.0->azure-cognitiveservices-vision-customvision) (1.15.0)

#Step 3
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
%matplotlib inline
    
import cv2
vidcap = cv2.VideoCapture('file:/Users/parag/Downloads/Detection.mp4')
success,image = vidcap.read()
 
count = 0
credentials = ApiKeyCredentials(in_headers={"Prediction-key":cv_key})
predictor = CustomVisionPredictionClient(endpoint=cv_endpoint, credentials=credentials)
 
while success:
    cv2.imwrite("frame%d.jpg" % count, image)
    success,image = vidcap.read()
    test_img_file = os.path.join('frame'+str(count)+'.jpg')
    test_img = Image.open(test_img_file)
    test_img_h, test_img_w, test_img_ch = np.array(test_img).shape
    print('Read a new frame: ', success)
count += 1
with open(test_img_file, mode="rb") as test_data:
    results = predictor.detect_image(project_id, model_name, test_data)

fig = plt.figure(figsize=(8, 8))
plt.axis('off')
draw = ImageDraw.Draw(test_img)
lineWidth = int(np.array(test_img).shape[1]/100)
object_colors = {
    "mask": "lightgreen",
    "no_mask": "yellow"
}

for prediction in results.predictions:
    color = 'white'# default for 'other' object tags
    if (prediction.probability*100) > 50:
        if prediction.tag_name in object_colors:
            color = object_colors[prediction.tag_name]
            left = prediction.bounding_box.left * test_img_w
            top = prediction.bounding_box.top * test_img_h
            height = prediction.bounding_box.height * test_img_h
            width = prediction.bounding_box.width * test_img_w
            points = ((left,top), (left+width,top), (left+width,top+height), (left,top+height),(left,top))
            draw.line(points, fill=color, width=lineWidth)
            plt.annotate(prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100),(left,top), backgroundcolor=color)
plt.imshow(test_img)
