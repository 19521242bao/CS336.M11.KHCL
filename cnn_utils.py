import os
import sys
from typing import List
import tarfile
import numpy as np
import cv2
import re

def signature_bit(data, projections):
  """
  LSH signature generation using random projection
  Returns the signature bits for two data points.
  The signature bits of the two points are different
  only for the plane that divides the two points.
  """
  sig = 0
  for p in projections:
    sig <<=  1
    if np.dot(data, p) >= 0:
      sig |= 1
  return sig
def MakeDirWithChecked(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def load_list(fname: str):
    """Plain text list loader. Reads from file separated by newlines, and returns a
    list of the file with whitespaces stripped.
    Args:
        fname (str): Name of file to be read.
    Returns:
        List[str]: A stripped list of strings, using newlines as a seperator from file.
    """

    return [e.strip() for e in open(fname, 'r').readlines()]


def extract(tar_url, extract_path='.'):
    # print tar_url
    tar = tarfile.open(tar_url, 'r')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract(item.name, "./" + item.name[:item.name.rfind('/')])
try:

    extract(sys.argv[1] + '.tgz')
    # print 'Done.'
except:
    name = os.path.basename(sys.argv[0])
    # print name[:name.rfind('.')], '<filename>'


#query_file_name = "radcliffe_camera_1_query.txt"
def cropImage(query_file_name: str, groundtruthFolderDir: str, imgFolderDir: str, outputFolderDir: str):
  print(query_file_name)
  query_key = query_file_name.replace("_query.txt", "")
  query_detail = load_list(groundtruthFolderDir + "/" + query_file_name)
  list = query_detail[0].split()
  imgName = list[0].replace("oxc1_", "")
  img = cv2.imread(imgFolderDir + "/%s.jpg" % imgName);
  x1 = int(float(list[1]))
  y1 = int(float(list[2]))
  x2 = int(float(list[3]))
  y2 = int(float(list[4]))
  crop_img = img[y1:y2, x1:x2]
  cv2.imwrite(outputFolderDir +"/%s.jpg" % query_key, crop_img)



# folder = evalGroundTruth
# names = []
# for basename in os.listdir(folder):
#   name = os.path.splitext(basename)[0];
#   matches = re.findall(".+_query", name)
#   if len(matches) > 0:
#     names.append(name)

# for name in names:
#   fileName = name + ".txt"
#   print(fileName)
#   cropImage(fileName, folder, datasetDir, evalQueryImg)
