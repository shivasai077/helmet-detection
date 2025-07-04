# OpenCV program to detect face in real time 
# import libraries of python OpenCV  
# where its functionality resides 
import cv2
import winsound
fre=400
dur=2000
  
# load the required trained XML classifiers 



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
  

# Trained XML file fosr detecting eyes 
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')  
  
# capture frames from a camera 
cap = cv2.VideoCapture(0) 
  
# loop runs if capturing has been initialized. 
while 1:  
  
    # reads frames from a camera 
    ret, img = cap.read()
    cv2.imshow('color output',img)
  
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray output',gray)
  
    # Detects faces of different sizes in the input image 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
  
    for (x,y,w,h) in faces: 
        # To draw a rectangle in a face  
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w]
        cv2.putText(img, 'face detected',(20,20),cv2.FONT_HERSHEY_SIMPLEX, 
				1, (0,0,255), 2, cv2.LINE_AA)
        print("Alert! face detected")
        #winsound.Beep(fre,dur)
        cv2.imwrite('face.jpg',img)
  
        # Detects eyes of different sizes in the input image 
        eyes = eye_cascade.detectMultiScale(roi_gray)  
  
        #To draw a rectangle in eyes 
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
            cv2.putText(roi_color, 'eyes detected',(20,20),cv2.FONT_HERSHEY_SIMPLEX, 
				1, (0,255,255), 2, cv2.LINE_AA)
  
    # Display an image in a window 
    cv2.imshow('img',img) 
  
    # Wait for Esc key to stop 
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
  
# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()
