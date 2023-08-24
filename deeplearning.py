

#import libraries
import numpy as np  
import cv2 
import tensorflow as tf
from scipy.special import softmax # to make the output of the model into a probability distribution that the sum of all the predictions =1

# load the face mask reconition model
import keras 
model = keras.models.load_model('model.keras') # loading the model

# load the pre-trained model (face detection model - Caffe model)
face_detection_model = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt','./models/res10_300x300_ssd_iter_140000_fp16.caffemodel') 

# label
labels = ['Mask', 'No Mask', 'Covered  Mouth chin' , 'Covered Nose mouth']  # the labels of the model

#associate each class with a color
def getColor(label):
    if label == "Mask":
        color = (0, 255, 0) #green
    elif label == "No Mask":
        color = (0, 0, 255) #red
    elif label == "Covered  Mouth chin":
        color = (0, 255, 255) #yellow
    elif label == "Covered Nose mouth":
        color = (255, 255, 0) #blue
    return color



#----------------------------------------------------RECOGNITION PART---------------------------------------------------------------------------------------------

#------------------Step 1: Face Detection--------------------------------------------------------------------------------------------------

#taking the blob from image, and pass it to the face detection model and get the bounding box of all the bolb their confidence score is greater than 0.5
#finally we drow the bounding box on the image and show it

def face_mask_prediction(img):
    image = img.copy()
    h,w = image.shape[:2] # taking the first two values of the shape
    blob = cv2.dnn.blobFromImage(image, 1.0, (300,300), (104.0, 117.0, 123.0) ,swapRB=True) # blobFromImage(image, scalefactor, size, mean, swapRB=True)

    face_detection_model.setInput(blob) # setting the blob as input to the network
    detections = face_detection_model.forward() # getting the detections from the network- in this detection there is the bounding box and the confidence score

    for i in range(0, detections.shape[2]): # looping over the detections
        confidence = detections[0,0,i,2] # extracting the confidence score
        if confidence > 0.5: # if the confidence score is greater than 0.5
            box = detections[0,0,i,3:7] * np.array([w,h,w,h]) # getting the coordinates of the bounding box, multiplying by the width and height of the image to normalize the coordinates
            box = box.astype('int') # converting the coordinates to integers
            pt1 = (box[0], box[1]) # getting the top left corner
            pt2 = (box[2], box[3]) # getting the bottom right corner
            #cv2.rectangle(image, pt1, pt2, (0,255,0), 2) # drawing the rectangle on the image

            
            #------------------Step 2: Data Preprocessing--------------------------------------------------------------------------------------------------
            
            #we will do the same data preprocessing as in the firt part.
            #crop the face, calculating the blob from image' ,flippiing the image ,resize,squize' rotate' flip , remove the negative value and normalaize.


            #crop the face
            face = img[box[1]:box[3], box[0]:box[2]] # cropping the face
            #compute the blob for the face
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (100,100), (104.0, 117.0, 123.0) ,swapRB=True) # blobFromImage(image, scalefactor, size, mean, swapRB=True)
            #the blob outpot is in 4D so we are reduce it to 3D
            face_blob_squeeze = np.squeeze(face_blob).T # removing the extra dimension,and transposing the image
            face_blob_rotate = cv2.rotate(face_blob_squeeze,cv2.ROTATE_90_CLOCKWISE) # rotating the image 90 degrees clockwise
            face_blob_flip = cv2.flip(face_blob_rotate,1) # flipping the image horizontally

            # normalize the image
            img_norm = np.maximum(face_blob_flip, 0) / face_blob_flip.max() # normalizing the image

            
            #----------------Step 3: Get the Prediction from the deep learning model--------------------------------------------------------------------------------------------------
           
            #we will pass the output of the data preperation to the cnn model and get the prediction.
            #the model get 4D image, but our image is 3D [100(pixels),100(pixels),3(rgb)] so we will add one more dimension to the image and get [1,100,100,3]
            # reshape the image


            img_input = img_norm.reshape(1,100,100,3) # reshaping the image to the input shape of the model

            # predict the image
            result= model.predict(img_input) # predicting the image
            result= softmax(result) # making the output of the model into a probability distribution that the sum of all the predictions =1
            #print(result)
            confidence_index=result.argmax() # getting the index of the maximum value in the result array
            confidence_score = result[0][confidence_index] # getting the confidence score of the prediction
            label = labels[confidence_index] # getting the label of the prediction
            label_text = "{}: {:.2f}%".format(label, confidence_score * 100) # formatting the label and the confidence score
            # get the color of the label
            color = getColor(label)
            cv2.rectangle(image, pt1, pt2, color, 2) # drawing the rectangle on the image
            cv2.putText(image, label_text,pt1,cv2.FONT_HERSHEY_PLAIN, 2, color, 2) # putting the label on the image

    return image # returning the image with the bounding box and the label

