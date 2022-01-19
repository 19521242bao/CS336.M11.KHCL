import numpy as np
import os
from utils import load_list
import argparse


def prepareValidationMaterial(grouthtruthUrl, rankListUrl, query):
  ranked_list = load_list(rankListUrl)
  pos_set = list(set(load_list(grouthtruthUrl + "/%s_good.txt" % query) + load_list(grouthtruthUrl + "/%s_ok.txt" % query)))
  junk_set = load_list(grouthtruthUrl + "/%s_junk.txt" % query)

  return ranked_list, pos_set, junk_set

def compute_ap(pos: List[str], amb: List[str], ranked_list: List[str]):
    """Compute average precision against a retrieved list of images. There are some bits that
    could be improved in this, but is a line-to-line port of the original C++ benchmark code.
    Args:
        pos (List[str]): List of positive samples. This is normally a conjugation of
        the good and ok samples in the ground truth data.
        amb (List[str]): List of junk samples. This is normally the junk samples in
        the ground truth data. Omitting this makes no difference in the AP.
        ranked_list (List[str]): List of retrieved images from query to be evaluated.
    Returns:
        float: Average precision against ground truth - range from 0.0 (worst) to 1.0 (best).
    """

    intersect_size, old_recall, ap = 0.0, 0.0, 0.0
    old_precision, j = 1.0, 1.0

    for e in ranked_list:
        if e in amb:
            continue

        if e in pos:
            intersect_size += 1.0

        recall = intersect_size / len(pos)
        precision = intersect_size / j
        ap += (recall - old_recall) * ((old_precision + precision) / 2.0)

        old_recall = recall
        old_precision = precision
        j += 1.0

    return ap
def compute_ap_at_k(pos: List[str], amb: List[str], ranked_list: List[str], k):
    """Compute average precision against a retrieved list of images. There are some bits that
    could be improved in this, but is a line-to-line port of the original C++ benchmark code.
    Args:
        pos (List[str]): List of positive samples. This is normally a conjugation of
        the good and ok samples in the ground truth data.
        amb (List[str]): List of junk samples. This is normally the junk samples in
        the ground truth data. Omitting this makes no difference in the AP.
        ranked_list (List[str]): List of retrieved images from query to be evaluated.
    Returns:
        float: Average precision against ground truth - range from 0.0 (worst) to 1.0 (best).
    """

    intersect_size, old_recall, ap = 0.0, 0.0, 0.0
    old_precision, j = 1.0, 1.0

    for e in ranked_list:
      if j <= k:
        if e in amb:
            continue

        if e in pos:
            intersect_size += 1.0

        recall = intersect_size / len(pos)
        precision = intersect_size / j
        ap += (recall - old_recall) * ((old_precision + precision) / 2.0)

        old_recall = recall
        old_precision = precision
        j += 1.0

    return ap
def cal_mAP(evalRankList,evalGroundTruth):
  aps = []

  for query_img_path in os.listdir():
    if query_img_path != ".ipynb_checkpoints":
      #print(query_img_path)
      name = os.path.splitext(query_img_path)[0]
      resultRankFileDir = evalRankList + "/" + name + ".txt"
      ranked_list, pos_set, junk_set = prepareValidationMaterial(evalGroundTruth, resultRankFileDir, name)

      a = compute_ap(ranked_list, junk_set, pos_set)
      aps.append(a)
      #print(name)
      #print(a);

  return np.mean(aps)
def cal_mAP_at_k(k,evalRankList,evalGroundTruth):
  aps = []

  for query_img_path in os.listdir(evalRankList):
    if query_img_path != ".ipynb_checkpoints":
      #print(query_img_path)
      name = os.path.splitext(query_img_path)[0]
      resultRankFileDir = evalRankList + "/" + name + ".txt"
      ranked_list, pos_set, junk_set = prepareValidationMaterial(evalGroundTruth, resultRankFileDir, name)

      a = compute_ap_at_k(ranked_list, junk_set, pos_set, k)
      aps.append(a)
      #print(name)
      #print(a);
      
  return np.mean(aps)


mAP1 = cal_mAP_at_k(1)
mAP4 = cal_mAP_at_k(4)
mAP10 = cal_mAP_at_k(20)
mAP100 = cal_mAP_at_k(100)
mAP = cal_mAP()

print("mAP@1: \t\t %s" % mAP1)
print("mAP@4: \t\t %s" % mAP4)
print("mAP@10: \t %s" % mAP10)
print("mAP@100: \t %s" % mAP100)
print("mAP: \t\t %s" % mAP)

def args_parse():

    parser = argparse.ArgumentParser(description="Methods extract image.")
    parser.add_argument('-i', '--input_folder',  default=".\data\train",
                        help="The path of the input image folder.")
    parser.add_argument('-o', '--output_folder', default=".\feature\SIFT",
                        help="The path of the output feature folder")
    parser.add_argument('-m', '--method', default="SIFT",
                        help="Method to extrac feature. We can choose such as: sift, hog, vgg16....")
    parser.add_argument('-lsh', '--LSH', type = int, default = 0,
                        help="Use Locality-Sensitive Hasing " )
    # End default optional arguments

    return vars(parser.parse_args())