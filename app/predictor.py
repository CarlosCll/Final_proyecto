import os
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
from skimage.morphology import disk
import tensorflow_hub as hub
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class Predictor:
    def __init__(self, labels_path: str, model_path: str, labels_path_E: str, model_path_E: str, image_size: int):
        self.labels_path = labels_path
        self.labels_path_E = labels_path_E
        self.model_path = model_path
        self.model_path_E = model_path_E
        self.image_size = image_size

        with open(self.labels_path, "r") as f:
            self.labels1 = [line.strip() for line in f]

        with open(self.labels_path_E, "r") as f:
            self.labels2 = [line.strip() for line in f]
        
        self.model1 = tf.keras.models.load_model((self.model_path),
                custom_objects={'KerasLayer':hub.KerasLayer}
                )
        self.model2 = tf.keras.models.load_model((self.model_path_E),
                custom_objects={'KerasLayer':hub.KerasLayer}
                )

    def predict_image(self, image: np.ndarray, opcion: int):
        I_HSV =  cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        I_HSV = np.double(I_HSV)
        [M,N,P]=I_HSV.shape 
        H= I_HSV[:,:,0] # COMPONENTE H
        S= I_HSV[:,:,1] # COMPONENTE S
        V= I_HSV[:,:,2] # COMPONENTE V
        umbral1=130
        umbral1_1=180
        #temp = ((H>umbral1 )& (H<umbral1_1)) | ((H>0) & (H<40))
        temp = (((H>0) & (H<30)))
        temp[temp]=1
        Z = np.uint8(temp)
        ret3, thresh3 = cv2.threshold(np.uint8(temp),0,1,cv2.THRESH_BINARY)
        contours_list, hierarchy = cv2.findContours(thresh3,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours_list) != 0:
            c = max(contours_list, key = cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            recorte_binario=np.expand_dims(thresh3, axis=2)[y:y + h,x: x + w,:]
            recorte_original=image[y:y + h,x: x + w,:]
        kernel = disk(5)
        Z2 = cv2.morphologyEx(recorte_binario, cv2.MORPH_CLOSE, kernel)
        Z2 = cv2.morphologyEx(Z2, cv2.MORPH_OPEN, kernel)
        I4_R= np.multiply(Z2,recorte_original[:,:,0])
        I4_G= np.multiply(Z2,recorte_original[:,:,1])
        I4_B= np.multiply(Z2,recorte_original[:,:,2])
        I4 = cv2.merge([I4_R,I4_G,I4_B])
        I4 =  np.uint8(I4)
        I6 =  cv2.resize(I4,(224,224), interpolation= cv2.INTER_LINEAR)
        # print(I6.shape)
        img = np.array(I6).astype(float)/255
        img = cv2.resize(img, (224,224))
        if (opcion == 0):
            pred = self.model1.predict(img.reshape(-1, 224, 224, 3))
            top_labels = {}
            top_labels_ids = np.flip(np.argsort(pred, axis=1)[0])
            for label_id in top_labels_ids:
                top_labels[self.labels1[label_id]] = pred[0,label_id].item()
            pred_label = self.labels1[np.argmax(pred)]
        else :
            pred = self.model2.predict(img.reshape(-1, 224, 224, 3))
            top_labels = {}
            top_labels_ids = np.flip(np.argsort(pred, axis=1)[0])
            for label_id in top_labels_ids:
                top_labels[self.labels2[label_id]] = pred[0,label_id].item()
            pred_label = self.labels2[np.argmax(pred)]
        
        return {'label': pred_label, 'top': top_labels}



    def predict_file(self, file, opcion: int):
        img = np.array(Image.open(file))
        # img = np.array(img).astype(float)/255
        return self.predict_image(img,opcion)
