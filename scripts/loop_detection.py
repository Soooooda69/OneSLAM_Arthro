import cv2
import numpy as np
from matplotlib import pyplot as plt
from datasets.dataset import ImageDataset
import os
from tqdm import tqdm

from DBoW.orb import ORB
from DBoW.R2D2 import R2D2
from DBoW.voc_tree import constructTree
from DBoW.matcher import *
import shutil
import pickle

r2d2 = R2D2()

def draw_keypoints(image, keypoints):
    for point in keypoints:
            (x,y) = int(point[0]), int(point[1])
            cv2.circle(image, (x,y), radius=1, color=(0, 255, 0), thickness=-1)
    return image

def r2d2_init(org_img_dict):

    K = 5 #classes of cluster
    L = 3 #depth of tree
    
    # Cluster the features beforehand, take it as a learning phase.
    if os.path.exists('./DBoW/r2d2_descriptors.pkl'):
        with open('./DBoW/r2d2_descriptors.pkl', 'rb') as file:
            image_descriptors = pickle.load(file)
    else:
        image_descriptors = r2d2.r2d2_features(org_img_dict)
        with open('./DBoW/r2d2_descriptors.pkl', 'wb') as file:
            pickle.dump(image_descriptors, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    # image_descriptors = R2D2.r2d2_features(org_img_dict)
    N = len(image_descriptors)      
    FEATS = []
    
    for feats in image_descriptors:
        FEATS += [np.array(fv, dtype='float32') for fv in feats]
    FEATS = np.vstack(FEATS)
    treeArray = constructTree(K, L, np.vstack(FEATS))
    tree = Tree(K, L, treeArray)
    # tree.build_tree(N, image_descriptors)
    matcher = Matcher(N, image_descriptors, tree)
    return tree, matcher


def r2d2_loop_detect(tree, matcher, image, idx):
    # idx: image frame index
    # image: first frames for samples
    num_points = 200
    # update the tree with descriptors detected from new image
    kps, des = r2d2.update_image(image, num_points)
    if des is not None:
        tree.update_tree(idx, des)
    
    #set threshold for similarity
    T = 0.88 
    res = {}
    # skip 100 frames to compare the similarity
    for j in range(tree.N-1):
        if abs(j - idx) < 100:
            continue
        print('Image {} vs Image {}: {}'.format(idx, j, matcher.cos_sim(tree.transform(idx), tree.transform(j))), end='\r')
        if matcher.cos_sim(tree.transform(idx), tree.transform(j)) >= T:
            res[j] = matcher.cos_sim(tree.transform(idx), tree.transform(j))
    if res:
        r = max(res.items(), key=lambda x:x[1])[0]
        print (f"Image {idx} is similar to Image {r} with similarity: ",res[r] )
        
        
def dbow_r2d2(org_img_dict, img_dict, save_path, num_points):
    kps_dict = {}
    tree, matcher = r2d2_init(org_img_dict)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        os.mkdir(save_path)
    T = 0.88 #threshold for similarity
    start_time = time.time()
    for i in img_dict.keys():
        
        kps, des = r2d2.update_image(img_dict[i], num_points)
        kps_dict[i] = kps
        if des is not None:
            tree.update_tree(i, des)

        res = {}
        for j in range(tree.N-1):
            if abs(j - i) < 100:
                continue
            print('Image {} vs Image {}: {}'.format(i, j, matcher.cos_sim(tree.transform(i), tree.transform(j))), end='\r')
            if matcher.cos_sim(tree.transform(i), tree.transform(j)) >= T:
                res[j] = matcher.cos_sim(tree.transform(i), tree.transform(j))
        if res:
            r = max(res.items(), key=lambda x:x[1])[0]
            print (f"Image {i} is similar to Image {r} with similarity: ",res[r] )
            # save loop frames to visualize result
            vis_loop_frames(img_dict, i, r, kps_dict, save_path)
    end_time = time.time()
    print('time (s): ', end_time - start_time, 'frames:', i)

def vis_loop_frames(img_dict, i, r, kps_dict, save_path):
    img1 = cv2.imread(str(img_dict[i]))
    img1 = draw_keypoints(img1, kps_dict[i])
    img2 = cv2.imread(str(img_dict[r]))
    img2 = draw_keypoints(img2, kps_dict[r])
    combined_image = np.concatenate((img1, img2), axis=1)
    save_img_pth = os.path.join(save_path, f'{i}vs{r}.png')
    cv2.imwrite(save_img_pth, combined_image)    

def remake_vis_dir(save_path):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        os.mkdir(save_path)

if __name__ == "__main__":
    
    orb = ORB()
    r2d2 = R2D2()
    # Load an image
    dataset = ImageDataset('../../datasets/sinus')
    org_img_dict = dataset.acquire_images()
    downsampled_keys = list(org_img_dict.keys())[:1000]
    img_dict = {key: org_img_dict[key] for key in downsampled_keys} 
    
    # dbow_orb(org_img_dict, img_dict, './temp_data/loop_detect/orb')
    # save detected images path '../../temp_data/loop_detect/sinus_r2d2'
    dbow_r2d2(org_img_dict, img_dict, '../../temp_data/loop_detect/sinus_r2d2', num_points=200)