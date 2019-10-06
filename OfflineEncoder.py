#!/usr/bin/python3

from os.path import join;
from math import ceil;
import numpy as np;
import cv2;
import tensorflow as tf;

class Encoder(object):

    def __init__(self, model_path = 'models'):

        self.model = tf.keras.models.load_model(join(model_path, 'vggface2.h5'), compile = False);

    def preprocess(self, img):

        assert img.shape[2] == 3;
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
        inputs = tf.expand_dims(img, axis = 0);
        outputs = tf.image.resize_with_pad(inputs, 224, 224);
        return outputs;

    def batch(self, imgs):

        inputs = [self.preprocess(img) for img in imgs];
        outputs = tf.concat(inputs, axis = 0);
        return outputs;

    def encode(self, imgs):

        assert type(imgs) is list;
        if len(imgs) == 0: return tf.zeros((0,self.model.outputs[0].shape[-1]), dtype = tf.float32);
        assert np.all([type(img) is np.ndarray and len(img.shape) == 3 for img in imgs]);
        batch = self.batch(imgs);
        return self.model(batch);

if __name__ == "__main__":

    assert tf.executing_eagerly() == True;
    encoder = Encoder();
