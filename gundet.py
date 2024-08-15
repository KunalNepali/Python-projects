import cv2
import imutils
import datetime
import matplotlib as plt 

gun_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(0)
firstFrame = None
gun_exist = False

while True:
    ret, frame = camera.read()
    if frame is None:
        break
    
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    guns = gun_cascade.detectMultiScale(gray, 1.3, 20, minSize=(100,100))
    
    if len(guns) > 0:
        gun_exist = True
        for (x, y, w, h) in guns:
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
    
    if firstFrame is None:
        firstFrame = gray
        continue
    
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S %p"), 
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    
    if gun_exist:
        print("Gun is detected")
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for plt.imshow
        plt.show()
        break
    else:
        cv2.imshow("Security Feed", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()