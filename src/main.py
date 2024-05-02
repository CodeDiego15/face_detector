import cv2

face_cascade = cv2.CascadeClassifier('src/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cv2.imwrite('src/img/detected_faces.jpg', image)
        
    cv2.imshow('image', image)
    c =  cv2.waitKey(30)
    
    if c == 27:
        break