import numpy as np
import cv2 
from keras.models import model_from_json
import pyautogui

class HandDetector():
    
    def __init__(self):
        self.bgSubtractor = cv2.createBackgroundSubtractorMOG2()
        self.MODEL_PATH = "./Models/GestureDetectionModel_2.json"
        self.MODEL_WEIGHT_PATH = "./Models/GestureDetectionModel_2_weights.h5"
        #self.gestures = {0: '01_palm', 1: '02_l',2: '03_fist',3: '04_fist_moved',4: '05_thumb',5: '06_index',6: '07_ok',7: '08_palm_moved',8: '09_c',9: '10_down'}
        self.gestures = {0: 'fist', 1: 'four', 2: 'l', 3: 'ok', 4: 'palm', 5: 'three', 6: 'two'}
    def load_model(self):
        
        json_file = open(self.MODEL_PATH, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        model = model_from_json(loaded_model_json)
        model.load_weights(self.MODEL_WEIGHT_PATH)
        print(":::Loaded model from disk")
        return model
        
    def create_dataset(self, mode, image, count):
        FILE_NAME = "./Dataset/Custom/" + mode + "/" + str(count) + ".png"
        cv2.imwrite(FILE_NAME, image)
        print ("::: IMAGE SAVED @", FILE_NAME )
        
        
       
    def start(self):
        count = 0
        cap = cv2.VideoCapture(0)
        frame_width = 300
        frame_height = 500
        model = self.load_model()
        
        while(cap.isOpened()):
            
            ret, frame = cap.read()
            cv2.rectangle(frame, (0, 0), (frame_width, frame_height) ,(0, 255, 0), 0)
            crop_image = frame[0:frame_height, 0: frame_width]
            '''
            blur = cv2.GaussianBlur(crop_image, (3,3), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            
            mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))
            fgmask = self.bgSubtractor.apply(hsv)
            kernel = np.ones((5, 5))
            
            dilation = cv2.dilate(fgmask, kernel, iterations=1)
            erosion = cv2.erode(dilation, kernel, iterations=1)
            
            filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
            ret, thresh = cv2.threshold(filtered, 127, 255, 0)
            '''
            extracted_hand = self.extract_hand(crop_image)
            grayMask = cv2.cvtColor(extracted_hand, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(grayMask, 127, 255, cv2.ADAPTIVE_THRESH_MEAN_C  )
            
            modelInput = cv2.resize(thresh, (320, 120))
            modelInput = np.expand_dims(modelInput, axis=-1)
            modelInput = np.expand_dims(modelInput, axis=0)
            pred = model.predict(modelInput)
            pred = np.argmax(pred[0])
            
            print(self.gestures[pred])
            if(pred == 4):
                pyautogui.press("space")
            
            cv2.imshow("Thresholded", thresh)
            cv2.imshow("layZ", frame)
            #cv2.putText(thresh, str(self.gestures[pred]),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
           
            k = cv2.waitKey(ord('q')) & 0xFF
              
            if k == ord('q'):
                break
            if k ==ord('a'):
                count += 1
                self.create_dataset("two",thresh, count)
            

            
        cap.release()
        cv2.destroyAllWindows()
        
   
    def extract_hand(self, frame):
        fgmask = self.bgSubtractor.apply(frame)
        kernel = np.ones((4, 4), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cv2.bitwise_and(frame, frame, mask=fgmask)
       
    

if __name__ == "__main__":
    detector= HandDetector()
    detector.start()

