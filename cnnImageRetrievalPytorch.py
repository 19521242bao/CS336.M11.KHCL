#import cac thu vien can thiet
from cirtorch.networks.imageretrievalnet import init_network
from cirtorch.networks.imageretrievalnet import extract_vectors
from cirtorch.networks.imageretrievalnet import extract_ss, extract_ms
from cirtorch.utils.general import get_data_root
from torch.utils.model_zoo import load_url
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from numpy.linalg import norm
from numpy import dot
import os, glob

def load_network():
    print("Loading Network ...")
    PRETRAINED = {
    'rSfM120k-tl-resnet50-gem-w'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'rSfM120k-tl-resnet101-gem-w'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'rSfM120k-tl-resnet152-gem-w'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet152-gem-w-f39cada.pth',
    'gl18-tl-resnet50-gem-w'            : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'gl18-tl-resnet101-gem-w'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
    'gl18-tl-resnet152-gem-w'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet152-gem-w-21278d5.pth',
    }

    state = load_url(PRETRAINED['gl18-tl-resnet152-gem-w'], model_dir=os.path.join(get_data_root(), 'networks'))
    net_params = {}
    net_params['architecture'] = state['meta']['architecture']
    net_params['pooling'] = state['meta']['pooling']
    net_params['local_whitening'] = state['meta'].get('local_whitening', False)
    net_params['regional'] = state['meta'].get('regional', False)
    net_params['whitening'] = state['meta'].get('whitening', False)
    net_params['mean'] = state['meta']['mean']
    net_params['std'] = state['meta']['std']
    net_params['pretrained'] = False
    # network initialization
    net = init_network(net_params)
    net.load_state_dict(state['state_dict'])

    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    ms = list(eval('[1, 2**(1/2), 1/2**(1/2)]'))

    print(">>> Sucess...")
    print('__________________________\n')
    return net, transform, ms

def Searching(feature_query, feature_corpus, top = 10):
    #compute cosine distance
    cosine_dis = {}
    for img in feature_corpus:
        cosine_dis[img] = dot(feature_corpus[img].T, feature_query)/(norm(feature_corpus[img].T)*norm(feature_query))
        
    results = sorted(cosine_dis.items(), key = lambda kv:(kv[1], kv[0]))
    return results[-top:]

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