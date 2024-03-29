import numpy as np
import cv2
from HoughTools import *
from KalmanTools import *

roi_defined = False
width=50
height=50
def define_ROI(event, x, y, flags, param):
    '''
    Definition region intérêt
    '''
    global r,c,w,h,roi_defined
    # if the left mouse button was clicked,
    # record the starting ROI coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        r, c = x, y
        roi_defined = False
    # if the left mouse button was released,
    # record the ROI coordinates and dimensions
    elif event == cv2.EVENT_LBUTTONUP:
        r2, c2 = x, y
        h = abs(r2-r)
        w = abs(c2-c)
        r = min(r,r2)
        c = min(c,c2)
        roi_defined = True


cap = cv2.VideoCapture(r'../Antoine_Mug.mp4')


###Selection de l'objet à tracker
# take first frame of the video
ret,frame = cap.read()
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)

while True:
    # display the image and wait for a keypress
    cv2.imshow("First image", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the ROI is defined, draw it!
    if (roi_defined):
        # draw a green rectangle around the region of interest
        cv2.rectangle(frame, (r,c), (r+h,c+w), (0, 255, 0), 2)
        #Coordonnée du centre :
        x_prec = r+h//2
        y_prec = c+w//2
        cv2.circle(frame, (x_prec,y_prec), 3, (0, 0, 255),2)

    # else reset the image..
    else:
        frame = clone.copy()
    # if the 'q' key is pressed, break from the loop
    if key == ord("q"):
        break

track_window = (r,c,h,w)
roi = frame[c:c+w, r:r+h]


#initialisation Kalman
vx=0
vy=0
X_prec = np.array([x_prec,y_prec,vx,vy])
P_prec= np.eye(4)

###Initialisation R-table


gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Calculate gradient magnitude and orientation
grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
grad_mag = np.sqrt(grad_x**2 + grad_y**2)
grad_ori = np.arctan2(grad_y, grad_x) * 180 / np.pi


# Apply threshold to gradient magnitude
threshold=80
_, mask = cv2.threshold(grad_mag, threshold, 255, cv2.THRESH_BINARY)

# Apply mask to orientations
grad_ori_masked = cv2.bitwise_and(grad_ori, grad_ori, mask=mask.astype(np.uint8))
gray_ori_masked = cv2.cvtColor(grad_ori_masked.astype(np.uint8),cv2.COLOR_GRAY2BGR)

# Show the orientations where masked pixels appear in red
grad_ori_display = cv2.applyColorMap(grad_ori_masked.astype(np.uint8), cv2.COLORMAP_HOT)
gray_ori_masked[mask == 0] = [0, 0, 255]
#normalization for display
grad_mag_normalized = grad_mag*255/np.max(grad_mag)
grad_ori_normalized = grad_ori*255/np.max(grad_ori)

#display image
cv2.imshow('Gradient Orientations', grad_ori_normalized.astype(np.uint8))

cv2.imshow('Gradient Norms', grad_mag_normalized.astype(np.uint8))

cv2.imshow('original', roi)

cv2.imshow('Selected orientations',gray_ori_masked)


#remplissage table
r_table = build_r_table(roi,(roi.shape[0]//2,roi.shape[1]//2))



#Coordonnée du centre


#Pour améliorer le modèle, tenir compte de l'angle!!! mais augmente temps de calcul de ouf.



###Tracking
while True:
    ret, frame = cap.read()
    if ret:

        clone = np.copy(frame)
        gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
        accumulator = accumulate_gradients(r_table,gray)
        #accumulator  = accumulate_gradients_Window(r_table,gray, x_prec, y_prec,30,30)

        acc = np.zeros_like(accumulator)
        x1 = max(0,x_prec - width//2)
        x2 = min(len(gray[0]),x_prec + width//2)
        y1 = max(0,y_prec - height//2)
        y2 = min(len(gray[1]),y_prec + height//2)
        cv2.rectangle(clone, (x1,y1), (x2,y2), (0, 255, 0), 2)
        acc[y1:y2+1, x1:x2+1] = accumulator[y1:y2+1, x1:x2+1] # fenêtre de proximité
        # m = n_max(accumulator, 10, x_prec, y_prec,30,30)
        m = n_max(acc, 10)
        y_points = [pt[1][0] for pt in m]
        x_points = [pt[1][1] for pt in m]

        #prendre le point avec le score le plus élevé
        i,j = np.unravel_index(acc.argmax(), acc.shape)
        distance = np.linalg.norm([x_prec-i,y_prec-j])

        #trouver le maximum le plus proche de la position précédente
        #print("=====================")
        print(x_prec,y_prec)
        x_prec_tampon = x_prec
        y_prec_tampon = y_prec
        for k in range(len(x_points)):
            di = np.linalg.norm([x_prec-x_points[k],y_prec-y_points[k]])
            #print("point : ",(x_points[k],y_points[k]), " dist : ", di)
            if di < distance:
                x_prec_tampon = x_points[k]
                y_prec_tampon = y_points[k]
                distance = di


        #Filtre de Kalman
        X_reel=np.array([j,i]) #mesure
        X_pred,P_pred = prediction(X_prec,P_prec) #prediction
        X_prec,P_prec = correction(X_reel,X_pred,P_pred)#correction

        # Actualisation des positions en fonction de la méthode choisie
        x_prec =  int(X_pred[0]) #j
        y_prec =  int(X_pred[1]) #i



        print(x_prec,y_prec)

        # for k in range(len(x_points)):
        #     cv2.circle(clone, (x_points[k],y_points[k]), 3, (0, 0, 255),2)
        cv2.circle(clone, (x_prec,y_prec), 3, (255, 0, 0),2)
        cv2.imshow("image base",clone)


        accumulator_norm = accumulator*255/np.max(accumulator)
        cv2.imshow("acc",acc.astype(np.uint8))


        k = cv2.waitKey(60) & 0xFF
        if k == 27:
            break
        elif k == ord('q'):
            break

    else:
        break

cv2.destroyAllWindows()
cap.release()
