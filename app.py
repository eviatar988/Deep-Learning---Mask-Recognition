#this is generic GUI app using PyQt5

from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
import numpy as np
import cv2 # OpenCV
import sys
from deeplearning import face_mask_prediction


class VideoCapture(qtc.QThread): # Video Capture Class , runs in a separate thread
    change_pixmap_signal = qtc.pyqtSignal(np.ndarray) # Signal to emit the image to the main thread
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
    
    def run(self):
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) # Open the camera using OpenCV - 0 is the default camera, CAP_DSHOW is needed to close the camera without errors on Windows
        while self._run_flag:
            ret, frame = cap.read() # Read an image from the camera (in BGR format)
            prediction_img = face_mask_prediction(frame) # Get the prediction image from the mask_detection function
            
            
            if ret: # If we have a valid image
                self.change_pixmap_signal.emit(prediction_img) # Emit the image to the main thread
        
        # Release the camera
        prediction_img = 127+np.zeros((450,600,3),dtype=np.uint8) # if we don't have a valid image we need to emit a black image
        self.change_pixmap_signal.emit(prediction_img) # Emit the image to the main thread
        cap.release()
            
    def stop(self): # Method to stop the thread
        self._run_flag = False # Set the run flag to False
        self.wait() # Wait for the thread to finish


class mainWindow(qtw.QWidget): # Main Window Class , MAIN THREAD
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mask Recognition")
        self.setWindowIcon(qtg.QIcon("icon.png"))
        self.setFixedSize(600, 600)
        
        #Qlabels
        self.label1 = qtw.QLabel("wallpaper",self)
        self.label1.setPixmap(qtg.QPixmap("images\wallpaper.jpg"))
        self.label2  = qtw.QLabel("<h3>Face Mask Recognition Application </h3>",self , alignment = qtc.Qt.AlignCenter)
        self.label_welcome = qtw.QLabel("<h2>Welcome</h2>",self , alignment = qtc.Qt.AlignCenter)
        
        
        #Qpushbutton
        self.button = qtw.QPushButton("Open Camera", self , clicked = self.buttonClicked , checkable = True)
        
        #layout
        vlayout = qtw.QVBoxLayout() # Vertical Layout
        
        vlayout.addWidget(self.label_welcome)
        vlayout.addWidget(self.label2) 
        vlayout.addWidget(self.button)
        vlayout.addWidget(self.label1) 

        
        #end
        self.setLayout(vlayout)
        self.show()
        
    def buttonClicked(self):
        print("Button Clicked")
        status = self.button.isChecked()
        if status: # If the button is toggled to True
            self.button.setText("Close Camera")
            
            #open camera - we are receiving the image from the video capture and passing it to the screen with "updateImage" function
            self.capture = VideoCapture() # Create a new VideoCapture object
            self.capture.change_pixmap_signal.connect(self.updateImage) # Connect the signal from the camera to the update_image slot
            self.capture.start() # Start the thread
        
        elif not status: # If the button is toggled to False
            self.button.setText("Open Camera")
            self.capture.stop() # Stop the thread
            
    
        
    @qtc.pyqtSlot(np.ndarray) # Slot to receive the image from the camera thread   
    def updateImage(self, img_array):
        rgb_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) # Convert the image from BGR to the RGB format
        h, w, ch = rgb_img.shape # Get the shape of the image
        bytes_per_line = ch * w # Get the number of bytes per line
        
        # Convert the image to the QImage format
        converted_img = qtg.QImage(rgb_img.data, w, h, bytes_per_line, qtg.QImage.Format_RGB888) # Convert the image to the RGB888 format
        scaled_img = converted_img.scaled(600, 600, qtc.Qt.KeepAspectRatio) # Scale the image to fit the label
        gt_img = qtg.QPixmap.fromImage(scaled_img) # Convert the image to a QPixmap
        
        #update to screen
        self.label1.setPixmap(gt_img) # Set the pixmap of the label to the QPixmap
        
        
        

if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    mw = mainWindow()
    sys.exit(app.exec_())