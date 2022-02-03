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
from cnn_config import SIZE_PROJECTION, RANDOM_SEED, DISTANCE_THRESOLD
from glob import glob
import numpy as np
from cnn_utils import signature_bit
import matplotlib.pyplot as plt


def save_result(rank, path_storage, score, query_name, save_file):

    np.savez_compressed(save_file, ranks=rank, paths=path_storage,
                        scores=score, query_name=query_name)


def retrieve_img_resnet(img_path, features_storage, input_path):
    querys_features = []
    extractor = None
    extractor = DeepRESNET()
    img = cv2.imread(img_path)
    # print(img)
    query_feature = extractor.extract(img)

    measure = Cosine_Measure()
    dist = measure.compute_similarity(query_feature, features_storage)
    score = np.sort(dist)[::-1]
    rank = np.argsort(dist)[::-1]
    ranks_show = rank[:20]
    scores_show = score[:20]
    query_name = glob(input_path + '/*')
    query_img = cv2.imread(img_path)
    list_result = []
    for rank in ranks_show:
        list_result.append(query_name[rank])
    return list_result


def search(query_path):
    feature_path = "./static/features/feature/RESNET.npz"
    input_path = "./static/images/database_oxford"
    data = np.load(feature_path, allow_pickle=True)
    features_storage = data['features']
    results = retrieve_img_resnet(query_path, features_storage, input_path)
    results = [i.split("\\")[-1] for i in results]

    return_data = []

    for result in results:
        return_data.append([0, 0, "./static/images/database_oxford/" + result])

    return return_data


if __name__ == "__main__":
    print(search(r"C:\Users\PND280\Documents\GitHub\CS336_M11.KHCL\query_img\all_souls_2.jpg"))
