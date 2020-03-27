#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import load_model
from keras.preprocessing import image



class dogcat:
    def __init__(self,filename):
        self.filename =filename


    @property
    def predict(self):
        # load model
        model = load_model('model.h5')

        # summarize model
       # model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (28, 28),color_mode='grayscale')
        test_image = image.img_to_array(test_image)

        test_image = test_image / 255
        test_image = np.expand_dims(test_image, axis = 0)


        result = model.predict(test_image)
        print(result)

        result = np.argmax(result, axis=1)
        print(result)

        # if result[0][0] == 1:
        #     prediction = 'dog'
        #     return [{ "image" : prediction}]
        # else:
        #     prediction = 'cat'
        #     return [{ "image" : prediction}]

        return str(result.item(0))