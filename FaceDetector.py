import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
#body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
cap = cv2.VideoCapture(0)

while True:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray)

	for (x, y, w, h) in faces:
		cv2.rectangle(img,(x, y),(x+w, y+h),(255, 0, 0), 2)
		#font = cv2.FONT_HERSHEY_SIMPLEX
		#cv2.putText(img,'Ivan',(x+w/2,y+h/2),font,0.5,(0,0,255),1,cv2.LINE_AA) # puts green "Ivan" around me
		print(x, y)

		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		# smiles = smile_cascade.detectMultiScale(roi_gray)

		for(ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2) # puts a green rectangle around eyes
			print(ex,ey)

		#for(sx,sy,sw,sh) in smiles:
			# cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0,0,255), 2) # puts a blue rectangle around smile
			# print(sx, sy)
            
	cv2.imshow('img', img)
	k = cv2.waitKey(30) & 0xFF
	if k == 32:
		break

cap.release()
cv2.destroyAllWindows()
