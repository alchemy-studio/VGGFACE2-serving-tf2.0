#!/usr/bin/python3

from os import listdir;
from os.path import join, exists, isdir;
import pickle;
import cv2;
import tensorflow as tf;
from OfflineDetector import Detector;
from OfflineEncoder import Encoder;

class Recognizer(object):

    def __init__(self, model_path = 'models'):

        self.detector = Detector(model_path);
        self.encoder = Encoder(model_path);
        self.knn = None;
        self.ids = None;

    def load(self):

        self.knn = cv2.ml.KNearest_create();
        self.knn.load('knn.xml');
        with open('ids.pkl', 'rb') as f:
            self.ids = pickle.loads(f.read());

    def loadFaceDB(self, directory = 'facedb'):

        if False == exists(directory) or False == isdir(directory):
            print("invalid directory");
            return False;
        self.ids = dict();
        count = 0;
        features = tf.zeros((0,8631), dtype = tf.float32);
        labels = tf.zeros((0,1), dtype = tf.int32);
        for f in listdir(directory):
            if isdir(join(directory,f)):
                imgs = list();
                label = count;
                self.ids[label] = f;
                count += 1;
                # only visit directory under given directory
                for img in listdir(join(directory,f)):
                    if False == isdir(join(directory,f,img)):
                        # visit every image under directory
                        image = cv2.imread(join(directory,f,img));
                        if image is None:
                            print("can't open file " + join(directory,f,img));
                            continue;
                        rectangles = self.detector.detect(image);
                        if rectangles.shape[0] != 1:
                            print("can't detect single face in image " + join(directory,f,img));
                            continue;
                        # crop square from facial area
                        upperleft = rectangles[0,0:2];
                        downright = rectangles[0,2:4];
                        wh = downright - upperleft;
                        length = tf.math.reduce_max(wh, axis = -1).numpy();
                        center = (upperleft + downright) // 2;
                        upperleft = center - tf.constant([length,length], dtype = tf.float32) // 2;
                        downright = upperleft + tf.constant([length,length], dtype = tf.float32);
                        face = image[upperleft[1]:downright[1],upperleft[0]:downright[0],:];
                        imgs.append(face);
                feature = self.encoder.encode(imgs);
                label = tf.tile(tf.constant([[label]], dtype = tf.int32), (feature.shape[0],1));
                features = tf.concat([features,feature],axis = 0);
                labels = tf.concat([labels, label], axis = 0);
        features = features.numpy(); # features.shape = (n, 8631)
        labels = labels.numpy(); # labels.shape = (n, 1)
        # train KD-tree
        self.knn = cv2.ml.KNearest_create();
        self.knn.train(features, cv2.ml.ROW_SAMPLE, labels);
        self.knn.save('knn.xml');
        # save ids
        with open('ids.pkl', 'wb') as f:
            f.write(pickle.dumps(self.ids));
        return True;

    def recognize(self, image):

        assert image is not None;
        rectangles = self.detector.detect(image);
        upperleft = rectangles[...,0:2];
        downright = rectangles[...,2:4];
        wh = downright - upperleft;
        length = tf.math.reduce_max(wh, axis = -1);
        center = (upperleft + downright) // 2;
        upperleft = center - tf.stack([length,length], axis = -1) // 2;
        downright = upperleft + tf.stack([length, length], axis = -1);
        upperleft = tf.reverse(upperleft, axis = [1]); # in h,w order
        downright = tf.reverse(downright, axis = [1]); # in h,w order
        boxes = tf.concat([upperleft, downright], axis = -1) / tf.cast(tf.tile(image.shape[0:2], (2,)), dtype = tf.float32);
        image = tf.expand_dims(tf.cast(image, dtype = tf.float32), axis = 0);
        faces = tf.crop_and_resize(image, boxes, tf.zeros((boxes.shape[0],), dtype = tf.int32),(224,224));
        for face in faces:
            cv2.imshow('face', face.numpy().astype('uint8'));
            cv2.waitKey();
        features = self.encoder.encode(faces);
        ret, results, neighbours, dist = self.knn.findNearest(features.numpy(), k = 1);
        # TODO

if __name__ == "__main__":

    assert tf.executing_eagerly() == True;
    recognizer = Recognizer();
    recognizer.loadFaceDB();
