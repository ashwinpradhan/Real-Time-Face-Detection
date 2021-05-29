import cv2

trained_faces = cv2.CascadeClassifier(
    'haar-cascade-files-master/haarcascade_frontalface_default.xml'
    )

webcam = cv2.VideoCapture(0)

while webcam.isOpened() :

    video_status, frame = webcam.read()
    
    if video_status == True :
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        coordinates = trained_faces.detectMultiScale(gray_frame, scaleFactor=1.2)
        
        for (x, y, w, h) in coordinates :
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        cv2.imshow('Face Detection', frame)
        key = cv2.waitKey(1)
        
        if key == 113 or key == 81 :
            break
    else :
        break
webcam.release()
cv2.destroyAllWindows()