#! /usr/bin/python3
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import cv2
HEIGHT = 1052
WIDTH = 1914

def gen_csv(path):
    classes = np.loadtxt(path + '/classes.csv', skiprows=1, dtype=str, delimiter=',')
    labels = classes[:, 2].astype(np.uint8)
    files = glob('{}/data/trainval/*/*_bbox.bin'.format(path))
    files.sort()
    name = path + '/training_data_all.csv'
    bbox_fields = ['R[0]','R[1]','R[2]','t[0]','t[1]','t[2]','sz[0]','sz[1]','sz[2]','class_id','flag']
    with open(name, 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')

        writer.writerow(['guid/image', 'R[0]','R[1]','R[2]','t[0]','t[1]','t[2]','sz[0]','sz[1]','sz[2]','class_id','flag', 'label'])

        for file in files:
            guid = file.split('/')[-2]
            idx = file.split('/')[-1].replace('_bbox.bin', '')
            b_num = 1
            bbox = np.fromfile(file, dtype=np.float32)
            bbox = bbox.reshape([-1, 11])
            found_valid = False
            for b in bbox:
                #ignore_in_eval
                # if bool(b[-1]):
                #     continue
                # found_valid = True
                class_id = b[9].astype(np.uint8)
                label = labels[class_id]
                writer.writerow(['{}/{}_image_{}.jpg'.format(guid, idx,b_num), b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],b[10], label])
                b_num += 1
            # if not found_valid:
            #     label = 0


    print('Wrote report file `{}`'.format(name))

def get_proj_mtx(img_path):
    proj = np.fromfile(img_path.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
    proj.resize([3, 4])
    return proj

def get_cloud(img_path):
    xyz = np.fromfile(img_path.replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
    xyz = xyz.reshape([3, -1])
    uv = np.vstack([xyz, np.ones_like(xyz[0, :])])
    return uv

def rot(n):
    n = np.asarray(n).flatten()
    assert(n.size == 3)

    theta = np.linalg.norm(n)
    if theta:
        n /= theta
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

        return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K
    else:
        return np.identity(3)

def get_3d_bbox(b_box):
    size = b_box[6:9]
    p0 = -size / 2
    p1 = size / 2
    vertices = np.array([
                        [p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
                        [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
                        [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]
                        ])
    edges = np.array([
        [2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
        [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]
    ], dtype=np.uint8)

    R = rot(b_box[0:3])
    t = b_box[3:6]
    vertices = R.dot(vertices)
    vertices[0,:] += t[0]
    vertices[1,:] += t[1]
    vertices[2,:] += t[2]
    return vertices,edges

def get_2d_bbox(bbox_2d_v):
    min = np.argmin(bbox_2d_v, axis = 1)
    max = np.argmax(bbox_2d_v, axis = 1)
    xmin = bbox_2d_v[0,min[0]]
    ymin = bbox_2d_v[1,min[1]]
    xmax = bbox_2d_v[0,max[0]]
    ymax = bbox_2d_v[1,max[1]]
    vertices = np.array([[xmin,xmin,xmax,xmax],[ymin,ymax,ymin,ymax]], dtype = np.uint32)
    bbox_coordinates = np.array([xmin,ymin,xmax,ymax],dtype = np.uint32)
    return vertices, bbox_coordinates

def disp_image(img):
    fig1 = plt.figure(1, figsize=(16, 9))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.imshow(img)
    ax1.axis('scaled')
    fig1.tight_layout()
    return ax1

def get_roi(img,bbox_2d):
    a = bbox_2d[0,0]
    a = constrain(a, 0, 1914)
    b = bbox_2d[0,2]
    b = constrain(b, 0, 1914)
    c = bbox_2d[1,0]
    c = constrain(c, 0, 1052)
    d = bbox_2d[1,1]
    d = constrain(d, 0, 1052)
    cropped_img = img[c:d, a:b]
    return cropped_img

def get_mask(bbox_2d):
    a = bbox_2d[0,0]  #xmin
    a = constrain(a, 0, 1914)
    b = bbox_2d[0,2]  #xmax
    b = constrain(b, 0, 1914)
    c = bbox_2d[1,0]  #ymin
    c = constrain(c, 0, 1052)
    d = bbox_2d[1,1]  #ymax
    d = constrain(d, 0, 1052)
    mask = np.zeros(HEIGHT, WIDTH)
    mask[c:d, a:b] = 1
    return mask

def constrain(x, a, b):
    if x < a:
        x = a
    elif x > b:
        x = b
    return x
