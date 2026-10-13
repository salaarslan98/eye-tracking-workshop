import sys
import cv2
import numpy
import pandas
import sklearn
import matplotlib
import joblib
import mediapipe as mp

print("Python", sys.version.split()[0])
for module in (cv2, numpy, pandas, sklearn, matplotlib, joblib, mp):
    print(module.__name__, module.__version__)

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
print("Face mesh loaded OK")
face_mesh.close()