from ast import Raise
from xml.dom import NotFoundErr
import cv2
import sys
from numpy import array as _A
import mediapipe as mp
import cv2
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt
from skimage.transform import SimilarityTransform, PiecewiseAffineTransform, warp
import pandas as pd
import plotly.express as px
from matplotlib.widgets import Slider

MESH_TRI = pd.read_csv('images/mesh_triangles.csv', delimiter=' ')
# all_triangles = []
# for col in MESH_TRI.columns:
#     all_triangles.append(_A([MESH_TRI[col] == x for x in LEFT_CHICK]).max(axis=0))
#     all_triangles.append(_A([MESH_TRI[col] == x for x in RIGHT_CHICK]).max(axis=0))
# CHICKS_triangles = MESH_TRI[_A(all_triangles).max(axis=0)]
# CHICKS_verticies = np.unique(CHICKS_triangles).flatten()

mp_face_mesh = mp.solutions.face_mesh

eyes_and_nose = _A([0,1,5,150,112, 463, 359, 409, 185])
RIGHT_CHICK = _A([447,345, 346,347,366, 280] )
LEFT_CHICK = _A([227, 116, 117,118,93, 50])
CHICKS_verticies = np.hstack([RIGHT_CHICK, LEFT_CHICK])
XY, UV = 0,0 #chicks diff
TRIANGLES = _A([[0,1,4],
                [1,2,4],
                [4,5,2],
                [3,2,5]])



def get_landmarks(image):
    '''
    returns single face landmarks (the image should contain exactly 1 face)
    image: cv2 BGR image
    result: landmarks in 0-1 coordinates
    '''
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
    # landmarks[:,0] = 1-landmarks[:,0]
    # landmarks[:,1] = 1-landmarks[:,1]
    return landmarks

def draw_landmarks(image, landmarks):
    # huge = cv2.resize(image, (0, 0), fx=1, fy=1)
    H, W = image.shape[:2]
    color = (255, 255, 255)
    color_chick = (200,0,0)
    for i, landmark in enumerate(landmarks[:, :2]):
        p = tuple((landmark * ( W, H)).astype(np.int))
        if i in CHICKS_verticies:
            cv2.putText(image, f"{i}", p, 0, 1 , color_chick, 1)
        else:        
            cv2.putText(image, f"{i}", p, 0, 2 , color, 1)
    return image

def show_landmarks(image, landmarks):
    plt.figure()
    image = image[:,:,::-1].copy()
    plt.imshow(image)
    h,w, _ = image.shape
    plt.scatter(landmarks[:,0]*w, landmarks[:,1]*h, s=20)
    plt.show()


if __name__ == "__main__":

    img_before = cv2.imread("images/p1_before.jpg")
    img_after = cv2.imread("images/p1_after.jpg")

    lm_ref_1 = get_landmarks(img_before)
    lm_ref_2 = get_landmarks(img_after)

    show_landmarks(img_before, lm_ref_1[LEFT_CHICK])
    
    tform = SimilarityTransform()
    tform.estimate(lm_ref_1[eyes_and_nose], lm_ref_2[eyes_and_nose])
    lm_ref_2 = tform.inverse(lm_ref_2)    
    # now lm_ref_1 and lm_ref_2 are in the same coordinate frame

    MAG = 5
    XY = lm_ref_1[LEFT_CHICK]
    UV = (lm_ref_2[LEFT_CHICK] - XY)*MAG
    h,w,_ = img_before.shape
    plt.quiver(XY[:,0]*w, XY[:,1]*h, 
               UV[:,0]*w, -UV[:,1]*h, angles='xy', scale_units='xy', scale=1)

    for tri in TRIANGLES:    
        t1 = plt.Polygon(XY[tri,:2]*_A([w,h]), color='b', fill=False)
        plt.gca().add_patch(t1)
        
    plt.show()


    
    src = lm_ref_1[:,:2].copy() * _A((w,h))    
    dst = src.copy()
    dst[LEFT_CHICK] = dst[LEFT_CHICK] + UV[:,:2] * _A((-w,h))

    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)
    out = warp(img_before, tform, output_shape=(h,w))
    plt.imshow(out[:,:,::-1])

    plt.figure()
    res = cv2.addWeighted(img_before,  0.5, out.astype('uint8'), 0.5, 0)
    plt.imshow(res[:,:,::-1])


















## Graveyard
    # image = cv2.imread('images/face-free-png-image.png')
    # lmarks = get_landmarks(image)
    # image = draw_landmarks(image, lmarks)
    # cv2.imwrite("do_not_commit_this.png", image)

    # xyz = np.vstack([lm_ref_1, lm_ref_2])
    # ids = _A([1.2]*len(lm_ref_1) + [1.6]*len(lm_ref_2))
    # df = pd.DataFrame({'x': xyz[:,0], 'y':xyz[:,1], 'z':xyz[:,2], 
    #                    'color':ids, 
    #                    'size':[1]*len(xyz)})
    # fig = px.scatter_3d(df, x='x', y='y', z='z',size='size',
    #           color='color', opacity=0.7)
    # fig.show()

    # fig = go.Figure(data = go.Cone(
    #                     x=XY[:,0],
    #                     y=XY[:,1],
    #                     z=XY[:,2],
    #                     u=UV[:,0],
    #                     v=UV[:,1],
    #                     w=UV[:,2],
    #                     # colorscale='Blues',
    #                     sizemode="absolute",
    #                     sizeref=10))
    # fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.8),
    #                             camera_eye=dict(x=1.2, y=1.2, z=0.6)))
    # fig.show()



    # fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
    #                             mode='markers')])

    # fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
    #                                mode='markers')])


    # img_before = draw_landmarks(img_before, lm_ref_1)
    # img_after = draw_landmarks(img_after, lm_ref_2)
    # cv2.imshow('before', img_before)
    # cv2.imshow('after', img_after)
    # cv2.waitKey()


    # image = cv2.imread("images/face-free-png-image.png")

    # print("Read image with shape:", image.shape)
    # ref_landmarks_before = get_landmarks(image)    
    # ref_landmarks_after = get_landmarks(image)
    # landmarks = get_landmarks(image)
    # print(f"Found face and got {len(landmarks)} landmarks")
    
       
    # x = landmarks[:,0]
    # y = landmarks[:,1]
    # z = landmarks[:,2]

    # mesh = go.Mesh3d(x=x, y=y, z=z)    
    # fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z)])
    # fig.show()
    # fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
    #                                mode='markers')])
    
    # # fig.show()




    # # cv2.imshow('1', huge)
    # # cv2.waitKey()