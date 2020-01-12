#! /usr/bin/python3
from utilities import *
import cv2
dir = "~/spradeep/pradeep/umich/rob535/finalProject/task1/data/trainval/"
dir_rel = "./data/trainval/"
# gen_csv(".")
#load training data from csv generated using gen_csv()
tr_data = np.loadtxt('training_data_all.csv', skiprows=1, dtype=str, delimiter=',')

#Extract columns of the csv into numpy arrays
images = tr_data[:, 0] #Image paths
labels = tr_data[:,-1].astype(np.uint8) #ranging from [0,4]
class_id = tr_data[:,10].astype(np.float32).astype(np.uint32) #ranging from [1,23]
b_boxes = tr_data[:,1:10].astype(np.float32) #Bounding box info: Rotation,Translation,Size

#----Testing utility functions----#
IMG_SIZE = 224
img_idx = 1
#1.Projection Matrix
proj = get_proj_mtx('./data/trainval/' + images[img_idx]) #Get projection mtx of first image
# print(proj)

#2.Getting centroids of a vehicle from b_box info
# centroid = b_boxes[1,3:6]
# centroid = np.append(centroid,[img_idx])
# print(centroid)

#3.Getting point cloud of an image
# cloud = get_cloud('./data/trainval/' + images[0])
# print(cloud[:,0])

#4.Projecting the centroid onto the image
# projection = proj.dot(centroid)
# projection = projection / projection[-1]
# print(projection)

#5.Constructing rotation matrix from R[0],R[1],R[2]
# R = b_boxes[img_idx,0:3]
# print(rot(R))

#6.Make bounding box from bbox info
[bbox_3d_v,bbox_3d_e]  = get_3d_bbox(b_boxes[img_idx])
bbox_3d_v = np.vstack([bbox_3d_v, np.ones_like(bbox_3d_v[0,:])]) #homogenizing the 3d coordinates

#7.Projecting bbox to 2d
bbox_2d_v = proj.dot(bbox_3d_v)
bbox_2d_v /= bbox_2d_v[-1,:]

#8.Getting 2d bbox
V = get_2d_bbox(bbox_2d_v)

#9.displaying image
path = dir_rel + images[img_idx]
img = cv2.imread(path,1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_cropped = get_roi(img, V)
try:
    img_cropped = cv2.resize(img_cropped, (IMG_SIZE, IMG_SIZE))
except cv2.error as e:
    print("Image out of bounds")
    exit()
cv2.imwrite('image.jpg',img_cropped)
plt.imshow(img_cropped)
# img_handle = disp_image(img)
# img_handle.scatter(V[0,:],V[1,:],c="red",marker='x', s=1)
plt.show()

# print(bbox_2d_v)
