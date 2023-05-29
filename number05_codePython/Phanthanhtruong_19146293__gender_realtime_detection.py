###########################################
#	end project report for AI		#
#	Phan Thanh Truong				#
#	19146293					#
#################################


import matplotlib.pyplot as plt
from keras.utils import load_img
from keras.utils.image_utils import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2 
import cvlib
vat = {1: 'female',2:'male'}
# load model
model = load_model('gender_detection2.h5')
webcam = cv2.VideoCapture(1)
while True:
    status, test_image = webcam.read()
    face, confidence = cvlib.detect_face(test_image)
    # test_image = cv2.imread('R.jpeg')
    gray = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    i = 0
    for idx, f in enumerate(face):
        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        cv2.rectangle(test_image,(startX-30,startY-30),(endX+30,endY+30),(203,12,255),2)
        img = gray[(startY-30):(endY+30),(startX-30):(endX+30)]
        img = cv2.resize(img,(100,100))
        cv2.imwrite('img.jpg',img)
        print('shape2',img.shape)
        img = load_img('img.jpg',target_size=(100,100))
        img = img_to_array(img)
        img = img.reshape(1,100,100,3)
        img = img.astype('float64')
        img =img/255
        result  = np.argmax(model.predict(img),axis=1)
        text = str(vat[result[0]])
        cv2.putText(test_image,text,(startX,startY),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        cv2.imshow('out',test_image)
    # press "Q" to stop
    k = cv2.waitKey(1)
    if k == 27 or 0xFF == ord('q'):
        break
# release resources
webcam.release()
cv2.destroyAllWindows()

