
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import os, json, cv2, random


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
setup_logger()

from itertools import count


from google.colab.patches import cv2_imshow
from collections import defaultdict


cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library

# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
# here appears to be a "zoo" of detectron 2 models, I will check to see best
predictor = DefaultPredictor(cfg)

IMAGE_OUTPUT_PATH = os.path.join('.', 'image_output')


def get_occurences():
  occurences = defaultdict(list)

  for i, image in enumerate(sorted(os.listdir(IMAGE_OUTPUT_PATH))):
    if i > 20:  # testing
      break

    print(image)
    # (h, w, rgb channels) eg (720, 1280, 3)
    img = cv2.imread(os.path.join(IMAGE_OUTPUT_PATH, image))

    cv2_imshow(img)
    predictions = predictor(img)
    # Instances(num_instances, pred_boxes, scores (probabilities), pred_classes
    # {'instances': Instances(num_instances=3, image_height=480, image_width=854, fields=[pred_boxes: Boxes(tensor([[  0.0000, 399.5721, 854.0000, 480.0000],
    #        [  0.0000, 475.3178, 854.0000, 480.0000],
    #        [  0.0000, 302.4116, 854.0000, 480.0000]], device='cuda:0')), scores: tensor([1.0000, 1.0000, 1.0000], device='cuda:0'), pred_classes: tensor([62, 62, 62], device='cuda:0')])}

    v = Visualizer(img[:, :, ::-1])  # exclude alpha (transparency) from RGB

    # draws bounding boxes??
    out = v.draw_instance_predictions(predictions["instances"].to("cpu"))

    # get indices for predicted classes for current frame
    predictions = predictions['instances'].pred_classes
    print(predictions)

    #print(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    # if len(predictions) and predictions[0]: #[0]:
    #   # cv2_imshow(out.get_image()[:, :, ::-1])
    #   print('detected')
    #   print(out)

    # ignore object counts for now (3 cats same as 1 cat)
    uniq_labels = {pred.item() for pred in predictions}
    print(f'uniq {uniq}')
    
    for pred in uniq_labels:
      # get ranges for predicted class: List[Tuple[start_frame, end_frame]]
      # eg [[s1, f1], [s2, f2]]
        tuples = occurences[pred]  # [[s1, f1], [s2, f2]]
        if tuples: 
          last_frame = tuples[-1][-1]  # last frame in last range
          # print('last frame', last_frame)
          if last_frame == i - 1:  # continue from last frame
            tuples[-1][-1] += 1
            continue
        # otherwise, create new range
        occurences[pred].append([i, i])
  return occurences

      # better way in python 3.8:
      # if tuples := occurences[pred] and tuples[-1][-1] == i - 1:  # continue from last frame in ranges
      #   tuples[-1][-1] += 1
      # else:  # start new range
      #   occurences[pred].append([i, i])

print(get_occurences())