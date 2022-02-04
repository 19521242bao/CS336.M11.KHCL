
import glob
import os 
from PIL import Image
import numpy as np
from retrieval import retrieve_img_vgg16,retrieve_img_resnet
import os
import pickle
from PIL import Image
import numpy as np
import cv2
from scipy import spatial
import time
import glob
from tqdm import tqdm
image_folder="data/oxford5k_images/"
query_files = glob.glob(os.path.join("evaluation/groundtruth", "*_query.txt"))
query_files = [x.replace("\\", "/") for x in query_files]
#print(query_files)
K=30
res_file_path="result_vgg16.txt"
res_file = open(res_file_path, 'a+')
AP_score_lst = []
for file in query_files:
    gt_imgs = []
    with open(file.replace("query", "good"), 'r') as gt_fi:
        gt_imgs += [x.strip() for x in gt_fi.readlines()]

    with open(file.replace("query", "ok"), 'r') as gt_fi:
        gt_imgs += [x.strip() for x in gt_fi.readlines()]
    K = len(gt_imgs)
   #print(gt_imgs)
    print("read query image ")
    with open(file, 'r') as query_fi:
        img_info = query_fi.readlines()[0].strip().split(' ')
        img_name = img_info[0].replace("oxc1_", "") + ".jpg"
        x_min, y_min, x_max, y_max = [int(float(x)) for x in img_info[1:]]
        
        query_img = Image.open(image_folder + img_name)
        query_img = query_img.crop((x_min, y_min, x_max, y_max))
        #print("crop success")
        #print(query_img)
        relevant_imgs= retrieve_img_resnet(query_img, K)
        #print(relevant_imgs)
        relevant_jugdes = []
        
        for img_info in relevant_imgs:
            img_name =img_info.replace(".jpg", "")
            #print(img_name)
            if img_name in gt_imgs:
                relevant_jugdes.append(1)
            else:
                relevant_jugdes.append(0)
        if np.sum(np.array(relevant_jugdes)) == 0:
            AP_score_lst.append(0)
            AP_score = 0
        else:
            rel_judge_count = 0
            precision_score_lst = []
            for i in range(len(relevant_jugdes)):
                if relevant_jugdes[i] == 1:
                    rel_judge_count += 1
                    precision_score_lst.append(rel_judge_count / (i+1))   
            AP_score = np.mean(np.array(precision_score_lst))
            #print(AP_score)
            AP_score_lst.append(AP_score)
        line = f'- Query: {file.split("/")[-1]}'
        res_file.write(line + '\n')
        print(line)

        line = f'\tAverage Precision: {np.round(AP_score, 2)}'
        res_file.write(line + '\n')
        print(line)

    line = '\n----------COMPLETE THE SYSTEM EVALUATION---------'
    res_file.write(line + '\n')
    print(line)

    MAP = np.round(np.mean(np.array(AP_score_lst)), 2)
    line = f'Mean Average Precision of system: {MAP}'
    res_file.write(line + '\n')
    print(line)
