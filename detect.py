from __future__ import division

# Import Files
import argparse
import logging.config
import os
import time

import cv2
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from utils import cv_utils
from utils import tf_utils

logging.config.fileConfig('logging.ini')

# Tensorflow graph path
FROZEN_GRAPH_PATH = 'models/ssd_mobilenet_v1/frozen_inference_graph.pb'

# Threshold which decides whether to draw a bounding box or not
SCORE_THRESHOLD = 0.5

# Test image path
image_path = 'test.jpg'
# Output path where image will be stored.
output_dir = '.'


def main():
    # Read TensorFlow graph
    detection_graph = tf_utils.load_model(FROZEN_GRAPH_PATH)

    # Read image from the disk
    img = cv2.imread(image_path)

    with tf.Session(graph=detection_graph) as sess:

        # Note time
        tic = time.time()
        boxes = []

        detection_dict = tf_utils.run_inference_for_batch(
            np.expand_dims(img, axis=0), sess)
        
        # boxes: list of lists containing bounding box coordinates 
        # for each cone detected
        boxes = detection_dict['detection_boxes']
        boxes = boxes[np.any(boxes, axis=2)]

        # boxes_scores: list containing probability of each detected cone
        boxes_scores = detection_dict['detection_scores']
        boxes_scores = boxes_scores[np.nonzero(boxes_scores)]

        # Draw bounding box
        for box, score in zip(boxes, boxes_scores):
            # Check if probability is greater than threshold value
            # then only draw a bounding box
            if score > SCORE_THRESHOLD:
                ymin, xmin, ymax, xmax = box
                text = '{:.2f}'.format(score)
                cv_utils.add_rectangle_with_text(
                    img, ymin, xmin, ymax, xmax,
                    (255, 0, 0), text)

        # Note time
        toc = time.time()

        # Calculate processing time
        processing_time_ms = (toc - tic) * 1000
        logging.debug('Detected {} objects in {:.2f} ms'.format(
            len(boxes), processing_time_ms))

        # Store File
        input_image_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = '{}-detection.jpg'.format(input_image_filename)
        cv2.imwrite(os.path.join(output_dir, output_filename), img)


if __name__ == '__main__':
    main()
