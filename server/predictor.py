
#!pip install pyyaml==5.1
from server.get_occurences import get_occurences
import torch, torchvision

#print(torch.__version__, torch.cuda.is_available())
#!gcc --version

# install detectron2: (Colab has CUDA 10.1 + torch 1.7)
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
#!pip install torch==1.7
import torch
#assert torch.__version__.startswith("1.7")
#!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
# exit(0)  # After installation, you need to "restart runtime" in Colab. This line can also restart runtime

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import sys
import os
import argparse
#!sudo -H pip install --upgrade youtube-dl
import cv2
#!youtube-dl -F "https://www.youtube.com/watch?v=50Uf_T12OGY&ab_channel=JUtah"


VIDEO_NAME = get_video(video_name='petting_zoo', video_link='https://www.youtube.com/watch?v=AVPuJMtzrCw')
IMAGE_OUTPUT_PATH = os.path.join('.', 'image_output')

SAMPLING_RATE = 1  # in fps
FRAMES_PER_SAMPLE = 1000 // SAMPLING_RATE
from itertools import count


from google.colab.patches import cv2_imshow
from collections import defaultdict


NUM_CLASSES = 80

#DetectionCheckpointer(model).load(file_path_or_url)  # load a file, usually from cfg.MODEL.WEIGHTS
#from detectron2.modeling import build_model
#model = build_model(cfg)


#!pip install torchvision=='0.1.7'

# map indices to names
index_to_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get('thing_classes')

occurences = get_occurences()
occurences = {index_to_name(i): ranges for i, ranges in occurences.items()}
# for x in range(NUM_CLASSES):
#   occurences[possible_predictions[x]] = occurences[x]
#   del occurences[x]

import json
json_data = json.dumps(occurences, indent=2)

print(json_data)

image_folder = 'output'
video_name = './output.mp4'

# make sure inference saves as jpg
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
print(images)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)


from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_ak47_dicts("ak47_dataset/val")


#we need to save the frame locations of the detected objects
#
for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=ak47_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])

def get_frames(video_name, ):
    extractImages()
    occurences = get_occurences()
    print(occurences)