# import cac thu vien can thiet
from grpc import server
from cirtorch.networks.imageretrievalnet import extract_vectors
from tqdm import tqdm
from cnnImageRetrievalPytorch import Searching, load_network
import numpy as np
import glob


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
                path + 'static/features/feature_oxford_2/' + img[:-3] + 'npy')
            pbar.update(1)
    print(">>> Sucess...")
    print('__________________________\n')

    return feature_method_1

def method_1(query_path, bbx, feature_corpus, net, transform, ms):
    feature_query = extract_vectors(
        net, [query_path], 1024, transform, bbxs=[bbx], ms=ms)
    results = Searching(feature_query, feature_corpus, 20)

    return results.reverse()

def preload():
    path_corpus = ".\static\images\database_oxford\\"
    corpus = load_corpus(path_corpus)
    feature_corpus = load_features("", corpus)
    net, transform, ms = load_network()

    # net.cuda()
    # net.eval()

    return feature_corpus, net, transform, ms

def search(query_path, feature_corpus, net, transform, ms):
    query_path.replace("\\", "\\\\")
    

    # Extract Query
    # query_img = cv2.imread(query_path)
    feature_query = extract_vectors(net, [query_path], 1024, transform, ms=ms)
    results = Searching(feature_query, feature_corpus, top=10)

    return_data = []

    for result in results:
        return_data.append([0, 0, "./static/images/resized_oxford/" + result[0]])

    return return_data


if __name__ == '__main__':
    pass
