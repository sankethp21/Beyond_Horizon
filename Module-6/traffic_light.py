import numpy as np
import cv2
from keras.models import load_model

model=load_model('Module-6/model.h5')

def test_an_image(file_path):

    desired_dim=(32,32)
    img = cv2.imread(file_path)
    img_resized = cv2.resize(img, desired_dim, interpolation=cv2.INTER_LINEAR)
    img_ = np.expand_dims(np.array(img_resized), axis=0)
    predicted_state= np.argmax(model.predict([img_]), axis=1)[0]
    return predicted_state

def traffic_lig(file_path):
    states = ['red', 'yellow', 'green', 'off']
    i = test_an_image(file_path)
    print(states[i])
    return i
    

