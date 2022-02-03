# import cac thu vien can thiet
from cirtorch.networks.imageretrievalnet import init_network
from cirtorch.networks.imageretrievalnet import extract_vectors
from cirtorch.networks.imageretrievalnet import extract_ss, extract_ms
from cirtorch.utils.general import get_data_root
from torch.utils.model_zoo import load_url
from torchvision import transforms
from tqdm import tqdm
from cnnImageRetrievalPytorch import Searching, load_network
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt


def load_corpus(path):
    print("Loading Corpus ...")
    list_image = glob.glob(path + '*.jpg')
    # print(list_image)

    print(">>> Sucess...")
    print('__________________________\n')
    list_image = [i.split('\\')[-1] for i in list_image]
    return list_image


def load_features(path, corpus):

    feature_method_1 = {}
    print("Loading Feature method 1...")
    with tqdm(total=len(corpus)) as pbar:
        for img in corpus:
            feature_method_1[img] = np.load(
                path + 'static/images/feature_oxford_2/' + img[:-3] + 'npy')
            pbar.update(1)
    print(">>> Sucess...")
    print('__________________________\n')

    return feature_method_1


def load_methods(root):
    os.chdir(root)
    # Method 1
    net, transform, ms = load_network()
    # net.cuda()
    # net.eval()

    return net, transform, ms


def method_1(query_path, bbx, feature_corpus, net, transform, ms):
    feature_query = extract_vectors(
        net, [query_path], 1024, transform, bbxs=[bbx], ms=ms)
    results = Searching(feature_query, feature_corpus, 20)

    return results.reverse()


def search(query_path):
    query_path.replace("\\", "\\\\")
    net, transform, ms = load_network()
    # net.cuda()
    # net.eval()

    path_corpus = ".\static\images\database_oxford\\"
    corpus = load_corpus(path_corpus)
    feature_corpus = load_features("", corpus)

    # query_path = r'C:\Users\PND280\Documents\GitHub\CS336_M11.KHCL\static\images\database_oxford\all_souls_000006.jpg'

    # Extract Query
    # query_img = cv2.imread(query_path)
    feature_query = extract_vectors(net, [query_path], 1024, transform, ms=ms)
    results = Searching(feature_query, feature_corpus, top=10)

    return_data = []

    for result in results:
        return_data.append([0, 0, "./static/images/resized_oxford/" + result[0]])

    return return_data
    # print("Results top 10:")
    # print(results)
    # print("...............................................")
    # print("Continue retrieval ? (Enter 0 to exit)")
    # key = int(input())
    # for result in results:
    #     print(result)
    #     print(result[0])
    #     img = cv2.imread(os.path.join("static\images\database_oxford", result[0]))
    #     plt.imshow(img)
    #     plt.show()


if __name__ == '__main__':
    print(list(search(r"C:\Users\PND280\Documents\GitHub\CS336_M11.KHCL\query_img\all_souls_2.jpg")))