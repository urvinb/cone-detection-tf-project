from __future__ import division

import argparse
import logging.config
import os
import time

import cv2
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from utils import cv_utils
from utils import operations as ops
from utils import tf_utils

logging.config.fileConfig('logging.ini')

FROZEN_GRAPH_PATH = 'models/ssd_mobilenet_v1/frozen_inference_graph.pb'

SCORE_THRESHOLD = 0.5
NON_MAX_SUPPRESSION_THRESHOLD = 0.5


def ispath(path):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError('No such file or directory: ' + path)
    else:
        return path

image_path = 'test.jpg'
output_dir = '.'


def main():
    detection_graph = tf_utils.load_model(FROZEN_GRAPH_PATH)
    img = cv2.imread(args.image_path)

    with tf.Session(graph=detection_graph) as sess:

        tic = time.time()
        boxes = []

        detection_dict = tf_utils.run_inference_for_batch(
            np.expand_dims(img, axis=0), sess)
        boxes = detection_dict['detection_boxes']
        boxes = boxes[np.any(boxes, axis=2)]

        boxes_scores = detection_dict['detection_scores']
        boxes_scores = boxes_scores[np.nonzero(boxes_scores)]

        for box, score in zip(boxes, boxes_scores):
            if score > SCORE_THRESHOLD:
                ymin, xmin, ymax, xmax = box
                text = '{:.2f}'.format(score)
                cv_utils.add_rectangle_with_text(
                    img, ymin, xmin, ymax, xmax,
                    (255, 0, 0), text)

        toc = time.time()
        processing_time_ms = (toc - tic) * 1000
        logging.debug('Detected {} objects in {:.2f} ms'.format(
            len(boxes), processing_time_ms))

        input_image_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = '{}-detection.jpg'.format(input_image_filename)
        cv2.imwrite(os.path.join(output_dir, output_filename), img)


if __name__ == '__main__':
    main()
