from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, concatenate
from keras.layers import Dropout, BatchNormalization, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping
import cv2 
from skimage.feature import hog
import argparse
import numpy as np

from keras.preprocessing import image

class DeepRESNET:

    def __init__(self):
        self.model  = ResNet50(weights='imagenet', include_top=True)
        # self.model.summary()

    def extract(self, img ):
        img = cv2.resize(img, (224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        resnet50_feature = self.model.predict(img_data)

        return resnet50_feature.flatten()



def main(args):
  
    img = cv2.imread(args['input_path'])
    extrator = None
    extractor = DeepRESNET()
    feature = extractor.extract(img)
    feature2 = extractor.extract(img)
    # print("Shape feature: ", feature.shape)
    # print("Shape feature 2: ", feature2.shape)
def args_parser():

    parser = argparse.ArgumentParser(description="Methods extract image.")
    parser.add_argument('-i', '--input_path',  required=True,
                        help="The path of the input image.")
    return vars(parser.parse_args())

if __name__ == "__main__":

    args = args_parser()
    # End default optional arguments
    # Print info arguments
    # print("Extract feature from image.".upper().center(100))
    # print(str("-"*63).center(100))
    # print("|{:<30}:\n|{:<30}|".format("Image path", args['input_path']).center(100))
    # print(str("-"*63).center(100))

    main(args)  