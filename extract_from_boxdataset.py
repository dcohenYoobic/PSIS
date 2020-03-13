from pycocotools.coco import COCO
import skimage.io as io
import cv2
import json
import os
import multiprocessing
import functools
from joblib import Parallel, delayed
from functools import partial
import signal
import time
from PIL import Image, ImageFilter,ImageEnhance,ImageOps,ImageFile
import numpy as np
import pandas as pd 
from tqdm import tqdm
from glob import glob
def PIL2array1C(img):
    '''Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0])



def image_generation(ref_image_id, ref_id,ref_mask_area, ref_instance_size, ref_xmax, ref_ymax, ref_xmin, ref_ymin,
                    ref_mask_array, path_file):
    #category_name = "box"
    #ann2ann_list = []
    sec_file = path_file.split("/")[-1]
    ann2ann={}
    pro_id = str(sec_file.split('.')[0])
    pro_image_id = int(ann2img2[pro_id])
    if ref_image_id==pro_image_id: return None
    file_name = sec_file.split('.')[0]+'.pbm'
    mask = Image.open(path_file)
    x,y,w,h = annotations2[pro_image_id]['bbox']
    xmin = int(round(x))
    ymin = int(round(y))
    xmax = int(round(x + w))
    ymax = int(round(y + h))
    instance_size = (xmax - xmin) * (ymax - ymin)
    if instance_size != 0:
        ref2pro_ratio = float(ref_instance_size) / float(instance_size)
    else: ref2pro_ratio=0
    if xmax == xmin or ymax == ymin : return None
    mask = mask.crop((xmin,ymin,xmax,ymax))
    mask = mask.resize(((ref_xmax-ref_xmin),(ref_ymax-ref_ymin)), Image.ANTIALIAS)
    mask_array = 255-PIL2array1C(mask)
    mask_area = np.sum(mask_array==255)
    ssd = np.sum((ref_mask_array-mask_array)**2)
    if ref_mask_area != 0: ref_ssd2instance_ratio = float(ssd) / float(ref_mask_area)
    else: ref_ssd2instance_ratio = 1
    if mask_area != 0: pro_ssd2instance_ratio = float(ssd) / float(mask_area)
    else: pro_ssd2instance_ratio = 1
    print(ref_ssd2instance_ratio, pro_ssd2instance_ratio )
    if ref_ssd2instance_ratio > 0.3 or pro_ssd2instance_ratio > 0.3 : return None
    if ref2pro_ratio > 3 or ref2pro_ratio < 0.3: return None
    if ssd == 0 : return None
    ann2ann[ref_image_id]=pro_image_id
    #ann2ann[ref_image_id]=[pro_image_id, ref2pro_ratio, ref_ssd2instance_ratio, pro_ssd2instance_ratio]
    #ann2ann_list.append(ann2ann)
    #print('Finding satisfactory Annotation_a %s and Annotation_b %s for class %s'%(ref_id,pro_id,category_name))
    return ann2ann

    
INVERTED_MASK = 1
INSTANCE_FILE = "/home/ubuntu/new_masks/"#MSCOCO instance mask output file path, e.g.'MSCOCO/masks/'
INSTANCE_FILE2 = "/home/ubuntu/mask_box"
ANNOTATION_FILE = "/home/ubuntu/output_psis_new"#annotation pair file path for each category,e.g.'MSCOCO/PSIS/'
JSON_FILE= "/home/ubuntu/annotations_lowclasses.json"
JSON_FILE2= "/home/ubuntu/annotations.json"
MAPPING2 = "/home/ubuntu/mapping.json"#MSCOCO anntation json file path,e.g.'MSCOCO/annotations/instances_train2017.json'
MAPPING= "/home/ubuntu/mapping_lowclasses.json"
#/home/ubuntu/mapping.json"
try:
    os.makedirs(ANNOTATION_FILE)
except:
    pass
with open(JSON_FILE, 'r') as f:
    dataset_src = json.load(f)
with open(JSON_FILE2, 'r') as f:
    dataset_src2 = json.load(f)
with open(MAPPING, 'r') as f:
    ann2img = json.load(f)
with open(MAPPING2, 'r') as f:
    ann2img2 = json.load(f) 
annotations = {annotation['index_unique']:annotation for annotation in dataset_src['annotations']}
annotations2 = {annotation['index_unique']:annotation for annotation in dataset_src2['annotations']}


from itertools import combinations,product
list_results = []
list_pbm = glob("{}/*/*.pbm".format(INSTANCE_FILE))
list_pbm2 = glob("{}/*/*.pbm".format(INSTANCE_FILE2))
#f=open(os.path.join(ANNOTATION_FILE,category_name,'ann2ann.txt'),'a')
f=open(os.path.join(ANNOTATION_FILE,'ann2ann_new.txt'),'a')
for ind, path_file in enumerate(tqdm(list_pbm)):
    first_file = path_file.split("/")[-1]
    file_name = first_file.split('.')[0]+'.pbm'
    ref_id = str(first_file.split('.')[0])
    ref_image_id = int(ann2img[ref_id])
    ref_mask = Image.open(path_file)
    ref_x,ref_y,ref_w,ref_h = annotations[ref_image_id]['bbox']
    ref_xmin = int(round(ref_x))
    ref_ymin = int(round(ref_y))
    ref_xmax = int(round(ref_x + ref_w))
    ref_ymax = int(round(ref_y + ref_h))
    ref_instance_size = (ref_xmax - ref_xmin) * (ref_ymax - ref_ymin)
    if ref_xmax == ref_xmin or ref_ymax == ref_ymin : contimue
    ref_mask = ref_mask.crop((ref_xmin,ref_ymin,ref_xmax,ref_ymax))
    ref_mask_array = 255-PIL2array1C(ref_mask)
    ref_mask_area = np.sum(ref_mask_array==255)
    #image_gen = lambda sec_file: image_generation(ref_image_id, ref_id,ref_mask_area, ref_instance_size, ref_xmax, ref_ymax, ref_xmin, ref_ymin,ref_mask_array, sec_file)
    #image_gen(filenames[0])
    #results = Parallel(n_jobs = -1, verbose = 10, backend = "multiprocessing")(delayed(image_generation)(ref_image_id, ref_id,ref_mask_area, ref_instance_size, ref_xmax, ref_ymax, ref_xmin, ref_ymin,
    #ref_mask_array, path_sec_file)for path_sec_file in list_pbm[ind:])
    results = Parallel(n_jobs = -1, verbose = 10, backend = "multiprocessing")(delayed(image_generation)(ref_image_id, ref_id,ref_mask_area, ref_instance_size, ref_xmax, ref_ymax, ref_xmin, ref_ymin,
                        ref_mask_array, path_sec_file)for path_sec_file in list_pbm2)
    temp_res = [x for x in results if x is not None]
    for ann2ann in temp_res:
        f.write(str(ann2ann)+'\n')

#file_names = filenames
#paired = list(product(filenames, filenames))
#df = pd.DataFrame({"filename":filenames})
#df2 = pd.DataFrame({"filename":filenames})
#print(df.head())
#print(len(df))
#print(len(df2.join(df, how = "outer", on = "filename")))
#couple_images = set(map(frozenset, combinations(set(filenames), 2)))
#print(couple_images[0:10])
#image_generation(category_list[0])
#results = Parallel(n_jobs = -1, verbose = 10, backend = "multiprocessing")(delayed(image_generation)(i,j) for i,j in product(filenames, filenames))
print("Done")
#for k in list_results:
#for ann2ann in k:
#f.write(str(ann2ann)+'\n')
f.close()
print('Master: I\'m Done')
