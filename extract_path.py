import os
import re
import glob

def num_sort(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]

def get_resized_db_image_paths(destfolder='./static/images/database_paris'):
    db = (list(glob.iglob(os.path.join(destfolder, '*.[Jj][Pp][Gg]'))))
    db.sort(key=num_sort)
    return db


with open("database_paris.txt", "w+") as file:
    descs = get_resized_db_image_paths()
    for img in descs:
        file.writelines(img.replace("\\", "/") + "\n")