from ast import Raise
from xml.dom import NotFoundErr
import cv2
import sys
from numpy import array as _A
import mediapipe as mp
import cv2
import numpy as np

mp_face_mesh = mp.solutions.face_mesh


if __name__ == "__main__":
    if len(sys.argv)>1:
        image = cv2.imread(sys.argv[1])
    else:
        image = cv2.imread("images/face-free-png-image.png")
    print("Read image with shape:", image.shape)
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks or len(results.multi_face_landmarks) == 0:
            Raise(NotFoundErr("No face found in image"))
        for face_landmarks in results.multi_face_landmarks:
            landmarks = _A([(a.x, a.y, a.z) for a in face_landmarks.landmark])
    
    print(f"Found face and got {len(landmarks)} landmarks")
    huge = cv2.resize(image, (0, 0), fx=4, fy=4)
    H, W = huge.shape[:2]
    color = (0, 0, 0)
    for i, landmark in enumerate(landmarks[:, :2]):
        p = tuple((landmark * ( W, H)).astype(np.int))
        cv2.putText(huge, f"{i}", p, 0, 1 , color, 1)
    cv2.imwrite("do_not_commit_this.png", huge)