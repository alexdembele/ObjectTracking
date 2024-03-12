import numpy as np
import cv2

#Vidéo à 25 fps dt=1/25
#X =[x,y,v_x,v_y]
A = np.eye(4)
dt = 1/25
A[0,2] =dt
A[1,3]=dt
Q = np.diag([0.1, 0.1, 1, 1])
R = np.diag([0.1, 0.1])
H=np.array([[1,0,0,0],[0,1,0,0.]])

def prediction(X_prec,P_prec):
    X = A@ X_prec
    P = A@P_prec@A.T + Q
    return X,P

def correction(X_mes,X_pred,P_pred):
    S = R + H@P_pred@H.T
    K = P_pred@H.T@np.linalg.inv(S)
    I = X_mes - H@X_pred
    X = X_pred + K@I
    P = (np.eye(4) - K@H)@P_pred
    return X,P

