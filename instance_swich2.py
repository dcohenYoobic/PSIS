from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
import json
import random
from PIL import Image, ImageFilter,ImageEnhance,ImageOps,ImageFile
import os
import cv2
import re
import copy 
import progressbar

def PIL2array1C(img):
    '''Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0])
#Prepare Json file
Ann2ann_FILE="/home/ubuntu/output_psis/ann2ann_new.txt"#annotation pair file path,e.g.'MSCOCO/PSIS/Omega_uin.txt'
COCO_Json_File="/home/ubuntu/annotations_lowclasses.json"#MSCOCO anntation json file path,e.g.'MSCOCO/annotations/instances_train2017.json'
COCO_Json_File_Target="/home/ubuntu/annotations.json"#MSCOCO anntation json file path,e.g.'MSCOCO/annotations/instances_train2017.json'
PSIS_Json_File="/home/ubuntu/augment_annotations_new.json"#PSIS output anntation json file path,e.g.'MSCOCO/annotations/instances_psis.json'
Mask_Image_FILE = "/home/ubuntu/new_masks"#MSCOCO instance mask output file path, e.g.'MSCOCO/masks/'
Mask_Image_FILE_Target = "/home/ubuntu/mask_box"
COCO_Image_FILE = "/home/ubuntu/datasets/dataset/JPEGImages"#MSCOCO image file path,e.g.'MSCOCO/images/train2017'
PSIS_Image_FIle = "/home/ubuntu/datasets/dataset/lowclasses_JPEGImages"#PSIS image file path,e.g.'MSCOCO/images/psis'
fjson_coco = open(COCO_Json_File, 'r')
fjson_coco_target = open(COCO_Json_File_Target, 'r')
fjson_psis = open(PSIS_Json_File, 'w')
INVERTED_MASK = 1
ann_json_coco = json.load(fjson_coco)
ann_json_coco_target = json.load(fjson_coco_target)
#ann_json_psis = json.load(fjson_psis)
#Prepare Data structure
ann2img = {}
ann2img_target = {}
num2cls = {}
img2anns = {}
img2anns_target = {}
psis_image_list = []
psis_annotation_list = []
target_ann_list=[]
source_ann_list=[]
ann2ann={}
psis_img_id=0
psis_ann_id=0
ann_json_psis = {'info': ann_json_coco['info'], 'licenses': list(ann_json_coco['licenses']),'categories':list(ann_json_coco['categories'])}
images = {image['id']:image for image in ann_json_coco['images']}
images_target = {image['id']:image for image in ann_json_coco_target['images']}
annotations = {annotation['index_unique']:annotation for annotation in ann_json_coco['annotations']}
annotations_target = {annotation['index_unique']:annotation for annotation in ann_json_coco_target['annotations']}

for i in range(len(ann_json_coco['annotations'])):
    ann_list=[]
    ann2img[ann_json_coco['annotations'][i]['index_unique']] = ann_json_coco['annotations'][i]['image_id']
    if (ann_json_coco['annotations'][i]['image_id'] in img2anns) == 0:
        ann_list.append(ann_json_coco['annotations'][i]['index_unique'])
        img2anns[ann_json_coco['annotations'][i]['image_id']]=ann_list
    else:
        img2anns[ann_json_coco['annotations'][i]['image_id']].append(ann_json_coco['annotations'][i]['index_unique'])
for i in range(len(ann_json_coco_target['annotations'])):
    ann_list=[]
    ann2img_target[ann_json_coco_target['annotations'][i]['index_unique']] = ann_json_coco_target['annotations'][i]['image_id']
    if (ann_json_coco_target['annotations'][i]['image_id'] in img2anns_target) == 0:
        ann_list.append(ann_json_coco_target['annotations'][i]['index_unique'])
        img2anns_target[ann_json_coco_target['annotations'][i]['image_id']]=ann_list
    else:
        img2anns_target[ann_json_coco_target['annotations'][i]['image_id']].append(ann_json_coco_target['annotations'][i]['index_unique'])
print('Data Done')
#Read annotation switch file
with open(Ann2ann_FILE) as f:
    for line in f.readlines():
        line=line.strip("\n")
        source_ann_id=line.split(':')[0].split('{')[1]
        target_ann_id=line.split(':')[1].split('}')[0]
        source_ann_list.append(int(source_ann_id))
        target_ann_list.append(int(target_ann_id))
        #source_ann_list.append(int(target_ann_id))
        #target_ann_list.append(int(source_ann_id))
ann2ann=dict(list(zip(source_ann_list,target_ann_list)))
f.close()
print('Read Done')
#print(ann2img.keys())
#Swichting instance and update annotation file
bar=progressbar.ProgressBar()
for i in bar(list(range(len(source_ann_list)))):
    #prepare image and mask id
    src_ann_id=source_ann_list[i]
    src_img_id=ann2img[src_ann_id]
    tar_ann_id=target_ann_list[i]
    tar_img_id=ann2img_target[tar_ann_id]
    src_ann=annotations[src_ann_id]
    tar_ann=annotations[tar_ann_id]
    category_name_src= annotations[src_ann_id]['classname']
    #category_name=num2cls[annotations[src_ann_id]['category_id']]
    #prepare image and mask
    #src_img_name = (str(src_img_id).zfill(12)+'.jpg')
    #tar_img_name = (str(tar_img_id).zfill(12)+'.jpg')
    src_img_name = src_ann["filename"]+'.jpg'
    tar_img_name = tar_ann["filename"]+'.jpg'
    src_image = Image.open(os.path.join(COCO_Image_FILE,src_img_name))
    tar_image = Image.open(os.path.join(COCO_Image_FILE,tar_img_name))
    src_mask_name = src_ann["filename"]+ "_"+ str(src_ann["id"])+'.pbm'
    tar_mask_name = tar_ann["filename"]+ "_"+str(tar_ann["id"])+'.pbm'
    src_mask = Image.open(os.path.join(Mask_Image_FILE,category_name_src,src_mask_name))
    category_name_tar= annotations[tar_ann_id]['classname']
    tar_mask = Image.open(os.path.join(Mask_Image_FILE,category_name_tar,tar_mask_name))
    #prepare bbox coordinate
    src_x,src_y,src_w,src_h = annotations[src_ann_id]['bbox']
    src_xmin = int(round(src_x))
    src_ymin = int(round(src_y))
    src_xmax = int(round(src_x + src_w))
    src_ymax = int(round(src_y + src_h))
    tar_x,tar_y,tar_w,tar_h = annotations[tar_ann_id]['bbox']
    tar_xmin = int(round(tar_x))
    tar_ymin = int(round(tar_y))
    tar_xmax = int(round(tar_x + tar_w))
    tar_ymax = int(round(tar_y + tar_h))
    #prepare segm coordinate
    src_cenx = src_x+0.5*src_w
    src_ceny = src_y+0.5*src_h
    tar_cenx = tar_x+0.5*tar_w
    tar_ceny = tar_y+0.5*tar_h
    th_x_src2tar = src_x+0.5*src_w-tar_x-0.5*tar_w
    th_y_src2tar = src_y+0.5*src_h-tar_y-0.5*tar_h
    th_w_src2tar = src_w/tar_w
    th_h_src2tar = src_h/tar_h
    src_area = annotations[src_ann_id]['area']
    tar_area = annotations[tar_ann_id]['area']
    #crop and resize images and masks
    src_image_a = src_image
    tar_image_a = tar_image
    src_image = src_image.crop((src_xmin,src_ymin,src_xmax,src_ymax))
    src_image = src_image.resize(((tar_xmax-tar_xmin),(tar_ymax-tar_ymin)),Image.ANTIALIAS)
    src_mask  = src_mask.crop((src_xmin,src_ymin,src_xmax,src_ymax))
    src_mask  = src_mask.resize(((tar_xmax-tar_xmin),(tar_ymax-tar_ymin)),Image.ANTIALIAS)
    tar_image = tar_image.crop((tar_xmin,tar_ymin,tar_xmax,tar_ymax))
    tar_image = tar_image.resize(((src_xmax-src_xmin),(src_ymax-src_ymin)), Image.ANTIALIAS)
    tar_mask  = tar_mask.crop((tar_xmin,tar_ymin,tar_xmax,tar_ymax))
    tar_mask  = tar_mask.resize(((src_xmax-src_xmin),(src_ymax-src_ymin)), Image.ANTIALIAS)
    #generate images
    src_mask = Image.fromarray(255-PIL2array1C(src_mask))
    tar_mask = Image.fromarray(255-PIL2array1C(tar_mask))
    src_image_a.paste(tar_image,(src_xmin,src_ymin),Image.fromarray(cv2.GaussianBlur(PIL2array1C(tar_mask),(3,3),2)))
    tar_image_a.paste(src_image,(tar_xmin,tar_ymin),Image.fromarray(cv2.GaussianBlur(PIL2array1C(src_mask),(3,3),2)))
    #save images
    src_psis_id = psis_img_id
    #src_psis_name=str(src_psis_id).zfill(12)+'.jpg'
    src_psis_name="augment_{}_".format(src_psis_id) + src_img_name
    src_image_a.save(os.path.join(PSIS_Image_FIle,src_psis_name))
    psis_img_id+=1
    tar_psis_id = psis_img_id
    tar_psis_name = "augment_{}_".format(tar_psis_id) + tar_img_name 
    #tar_psis_name=str(tar_psis_id).zfill(12)+'.jpg'
    tar_image_a.save(os.path.join(PSIS_Image_FIle,tar_psis_name))
    psis_img_id+=1
    ##update annotation file
    src_ann_list = img2anns[src_img_id]
    tar_ann_list = img2anns_target[tar_img_id]
    for j in range(len(src_ann_list)):
        if src_ann_id == src_ann_list[j]:
            src_ann_bak = copy.deepcopy(src_ann)
            src_ann_bak['image_id'] = tar_psis_id
            src_ann_bak['id'] = psis_ann_id
            src_ann_bak["bbox"]=[tar_x,tar_y,tar_w,tar_h]
            src_ann_bak["filename"] = tar_psis_name.split(".")[0]
            src_ann_bak["classname"] = category_name_src
            if ann['iscrowd'] == 0:
                for k in range(len(src_ann_bak['segmentation'][0])):
                    if k%2==0:
                        src_ann_bak['segmentation'][0][k]=src_ann_bak['segmentation'][0][k]-th_x_src2tar
                        src_ann_bak['segmentation'][0][k]=round(tar_cenx+(src_ann_bak['segmentation'][0][k]-tar_cenx)/th_w_src2tar,2)
                    else:
                        src_ann_bak['segmentation'][0][k]=src_ann_bak['segmentation'][0][k]-th_y_src2tar
                        src_ann_bak['segmentation'][0][k]=round(tar_ceny+(src_ann_bak['segmentation'][0][k]-tar_ceny)/th_h_src2tar,2)
                src_ann_bak['area'] = tar_area / th_w_src2tar / th_h_src2tar
            psis_ann_id+=1
            psis_annotation_list.append(src_ann_bak)
        else:
            ann = copy.deepcopy(annotations[src_ann_list[j]])
            #ann['image_id'] = src_psis_id
            #ann['id'] = psis_ann_id
            #psis_ann_id+=1
            #psis_annotation_list.append(ann)
            #attached instance update
            x,y,w,h = annotations[src_ann_list[j]]['bbox']
            xmin = int(round(x))
            ymin = int(round(y))
            xmax = int(round(x + w))
            ymax = int(round(y + h))
            #if src_xmin<xmin and src_ymin<ymin and src_xmax>xmax and src_ymax>ymax:
            if False:
                ann = copy.deepcopy(annotations[src_ann_list[j]])
                ann['image_id'] = tar_psis_id
                ann['id'] = psis_ann_id
                #ann['bbox'] = [tar_x-src_x+xmin,tar_y-src_y+ymin,w*tar_w/src_w,h*tar_h/src_h]
                ann["filename"] = tar_psis_name.split(".")[0]
                if ann['iscrowd'] == 0:
                    for k in range(len(ann['segmentation'][0])):
                        if k%2==0:
                            ann['segmentation'][0][k]=ann['segmentation'][0][k]-th_x_src2tar
                            ann['segmentation'][0][k]=round(tar_cenx+(ann['segmentation'][0][k]-tar_cenx)/th_w_src2tar,2)
                        else:
                            ann['segmentation'][0][k]=ann['segmentation'][0][k]-th_y_src2tar
                            ann['segmentation'][0][k]=round(tar_ceny+(ann['segmentation'][0][k]-tar_ceny)/th_h_src2tar,2)
                    ann['area'] = ann['area'] / th_w_src2tar / th_h_src2tar
                psis_ann_id+=1
                psis_annotation_list.append(ann)
    src_img_ann = copy.deepcopy(images[src_img_id])
    src_img_ann['id'] = src_psis_id
    src_img_ann['filename'] =  src_psis_name
    psis_image_list.append(src_img_ann)
    for j in range(len(tar_ann_list)):
        if tar_ann_id == tar_ann_list[j]:
            tar_ann_bak = copy.deepcopy(tar_ann)
            tar_ann_bak['image_id'] = src_psis_id
            tar_ann_bak['id'] = psis_ann_id
            tar_ann_bak["bbox"] = [src_x, src_y, src_w, src_h]
            tar_ann_bak["filename"] = src_psis_name.split(".")[0]
            tar_ann_bak["classname"] = category_name_tar
            if ann['iscrowd'] == 0:
                for k in range(len(tar_ann_bak['segmentation'][0])):
                    if k%2==0:
                        tar_ann_bak['segmentation'][0][k]=tar_ann_bak['segmentation'][0][k]+th_x_src2tar
                        tar_ann_bak['segmentation'][0][k]=round(src_cenx+(tar_ann_bak['segmentation'][0][k]-src_cenx)*th_w_src2tar,2)     
                    else:
                        tar_ann_bak['segmentation'][0][k]=tar_ann_bak['segmentation'][0][k]+th_y_src2tar
                        tar_ann_bak['segmentation'][0][k]=round(src_ceny+(tar_ann_bak['segmentation'][0][k]-src_ceny)*th_h_src2tar,2)
                tar_ann_bak['area'] = tar_area * th_w_src2tar * th_h_src2tar
            psis_ann_id+=1
            psis_annotation_list.append(tar_ann_bak)
        else:
            ann = copy.deepcopy(annotations[tar_ann_list[j]])
            #ann['image_id'] = tar_psis_id
            #ann['id'] = psis_ann_id
            #psis_ann_id+=1
            #psis_annotation_list.append(ann)
            x,y,w,h = annotations[tar_ann_list[j]]['bbox']
            xmin = int(round(x))
            ymin = int(round(y))
            xmax = int(round(x + w))
            ymax = int(round(y + h))
            #attached instance update
            #if tar_xmin<xmin and tar_ymin<ymin and tar_xmax>xmax and tar_ymax>ymax:
            if False:
                ann = copy.deepcopy(annotations[tar_ann_list[j]])
                ann['image_id'] = src_psis_id
                ann['id'] = psis_ann_id
                #ann['bbox'] = [src_x-tar_x+xmin,src_y-tar_y+ymin,w*src_w/tar_w,h*src_h/tar_h]
                ann["filename"] = src_psis_name.split(".")[0]
                if ann['iscrowd'] == 0:
                    for k in range(len(ann['segmentation'][0])):
                        if k%2==0:
                            ann['segmentation'][0][k]=ann['segmentation'][0][k]+th_x_src2tar
                            ann['segmentation'][0][k]=round(src_cenx+(ann['segmentation'][0][k]-src_cenx)*th_w_src2tar,2)
                        else:
                            ann['segmentation'][0][k]=ann['segmentation'][0][k]+th_y_src2tar
                            ann['segmentation'][0][k]=round(src_ceny+(ann['segmentation'][0][k]-src_ceny)*th_h_src2tar,2)
                    ann['area'] = ann['area'] * th_w_src2tar * th_h_src2tar
                psis_ann_id+=1
                psis_annotation_list.append(ann)
    tar_img_ann = copy.deepcopy(images[tar_img_id])
    tar_img_ann['id'] = tar_psis_id
    tar_img_ann['filename'] =  tar_psis_name
    psis_image_list.append(tar_img_ann)
print('Switch Done')

ann_json_psis['images'] = psis_image_list
ann_json_psis['annotations'] = psis_annotation_list
json.dump(ann_json_psis,fjson_psis)
fjson_coco.close()
fjson_psis.close()
print('Master: I\'m Done')
