import os
import cv2
from numpy import extract
from tqdm import tqdm
from scipy.spatial import cKDTree
from skimage.measure import ransac
from skimage.transform import AffineTransform

from similarity_measure import *
from extraction.VGG16 import DeepVGG16
from extraction.RESNET import DeepRESNET
from config import SIZE_PROJECTION, RANDOM_SEED,DISTANCE_THRESOLD
from glob import glob
import numpy as np
from utils import signature_bit
import matplotlib.pyplot as plt

def save_result(rank,path_storage, score, query_name, save_file):
    
    np.savez_compressed(save_file, ranks = rank, paths = path_storage, scores = score, query_name = query_name)
def retrieve_img_resnet(img_path,features_storage,input_path ):
    querys_features = []
    extractor = None
    extractor = DeepRESNET()
    img = cv2.imread(img_path)
    #print(img)
    query_feature = extractor.extract(img)

    measure = Cosine_Measure()
    dist = measure.compute_similarity(query_feature, features_storage )
    score = np.sort(dist)[::-1]
    rank = np.argsort(dist)[::-1]
    ranks_show=rank[:20]
    scores_show=score[:20]
    query_name = glob(input_path + '/*')
    query_img=cv2.imread(img_path)
    list_result=[]
    for rank in ranks_show:
        list_result.append(query_name[rank])
    return list_result

def main():
    file_name="feature/RESNET.npz"
    img_path="data/oxford5k_images/ashmolean_000216.jpg"
    data = np.load(file_name,  allow_pickle=True)
    features_storage = data['features']
    score,rank=retrieve_img_resnet(img_path,features_storage)
    ranks_show=rank[:20]
    scores_show=score[:20]
    print(scores_show)
    input_path="data/oxford5k_images"
    query_name = glob(input_path + '/*')
    query_img=cv2.imread(img_path)
    plt.imshow(query_img)
    plt.show()
    for rank in ranks_show:
        img_result = cv2.imread(query_name[rank])
        plt.imshow(img_result)
        plt.show()
    print(ranks_show)
if __name__ == "__main__":
    main()