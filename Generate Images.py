from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from math import sqrt, ceil
import cv2
from matplotlib import pyplot
import io
import os
import pandas as pd 
import shutil
import string
import constants as const
# import helper_apis as apis
import numpy as np
from allpairspy import AllPairs
from PIL import Image, ImageOps
from numpy import asarray
from numpy import save
from numpy import load
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='raw_data', help='select from shear, rotation, translation_x, translation_y, brightness, contrast, zoom_x, zoom_y, zoom_xy, channel_shift_intensity, blur_avg, blur_median, txty')
parser.add_argument('--height', type=int, default=128, help='select height for image, max < 340')
parser.add_argument('--width', type=int, default=128, help='select width for image, max < 188')
parser.add_argument('--replace', type=bool, default=True, help='do you want to replace the pre-existing data files: Y: True or N: False')

args = parser.parse_args()

size = (args.width,args.height)
replace = args.replace
transformation_list = ['shear', 'rotation', 'translation_x', 'translation_y', 'brightness', 'contrast', 'zoom_x', 'zoom_y', 'zoom_xy', 'channel_shift_intensity', 'blur_avg', 'blur_median', 'txty']

while(1):
    if(args.dataset in transformation_list or args.dataset == 'raw_data' or args.dataset == "All"):
        break
    print("Enter Valid dataset name")
    sys.exit()

dataset_name = 'All'
if(args.dataset != 'All'):
    dataset_name = args.dataset


def generate_step_list(min_val, max_val, step):
    transformation = []
    i = min_val
    while(i <= max_val):
        transformation.append(i)
        i = i + step
    return transformation

datasets = ['shear', 'theta', 'tx', 'ty', 'brightness', 'contrast', 'zx', 'zy', 'zxzy', 'channel_shift_intensity', 'blur_avg', 'blur_median', 'txty']


shear_transformation = generate_step_list(-40,40,5)
theta_transformation = generate_step_list(-180,180,15)
tx_transformation = generate_step_list(-60,60,20)
txty_transformation = generate_step_list(-60,60,20)
brightness_transformation = generate_step_list(0,1.5,0.2)
contrast_transformation = generate_step_list(0,1.5,0.2)
zx_transformation = generate_step_list(0, 2.5, 0.5)
channel_shift_intensity_transformation = generate_step_list(1,15,1)
blur_avg_transformation = generate_step_list(5,50,5)
blur_median_transformation = generate_step_list(5,50,4)

transformation_names = [shear_transformation, theta_transformation, tx_transformation, tx_transformation, brightness_transformation, contrast_transformation, zx_transformation, zx_transformation, zx_transformation, channel_shift_intensity_transformation, blur_avg_transformation, blur_median_transformation, txty_transformation]

transformation_transformation_dict = dict(zip(datasets, transformation_names))


datagen = ImageDataGenerator()


def apply_contrast (data, transformation, count, param, number_of_times):
    newim = cv2.convertScaleAbs(data, alpha= transformation[number_of_times], beta=0)
    transformation_dataset[count] = cv2.resize(newim, (size[0],size[1]))

def apply_once (data, transformation, count, param, number_of_times):
    newim = datagen.apply_transform(x=data, transform_parameters={param: transformation[number_of_times]})
    transformation_dataset[count] = cv2.resize(newim, (size[0],size[1]))


def apply_txty (data, transformation, count, param, number_of_times):
    newim = datagen.apply_transform(x=data, transform_parameters={'tx': transformation[int(number_of_times/5)], 'ty':transformation[int(number_of_times%5)]})
    transformation_dataset[count] = cv2.resize(newim, (size[0],size[1]))

def apply_zxzy(data, transformation, count, param, number_of_times):
    newim = datagen.apply_transform(x=data, transform_parameters={'zx': transformation[number_of_times], 'zy': transformation[number_of_times]})
    transformation_dataset[count] = cv2.resize(newim, (size[0],size[1]))
    
def apply_blur (data, transformation, count, param, number_of_times):
    if(param == 'blur_median'):
        newim = cv2.medianBlur(data,transformation[number_of_times])
    else:
        newim = cv2.blur(data,(transformation[number_of_times],transformation[number_of_times]))
    transformation_dataset[count] = cv2.resize(newim, (size[0],size[1]))
        
def save_dataset(param, number_of_times):
    if(param != 'txty'):
        dataset_to_save = transformation_dataset[shuffling_list]
        data = asarray(dataset_to_save)
        print("Saving: ", str(param) + "_" + str(transformation_transformation_dict[param][number_of_times]))
        data_save_path = "flags/Modified_DataSets1/" + str(param) + "_" + str(transformation_transformation_dict[param][number_of_times]) + ".npy"
        save(data_save_path, data)
    else:
        dataset_to_save = transformation_dataset[shuffling_list]
        data = asarray(dataset_to_save)
        print("Saving: ", str(param) + "_" + str(transformation_transformation_dict[param][int(number_of_times/len(transformation_transformation_dict[param]))]) + "_" + str(transformation_transformation_dict[param][int(number_of_times%len(transformation_transformation_dict[param]))]))
        data_save_path = "flags/Modified_DataSets1/" + str(param) + "_" + str(transformation_transformation_dict[param][int(number_of_times/len(transformation_transformation_dict[param]))]) + "_" + str(transformation_transformation_dict[param][int(number_of_times%len(transformation_transformation_dict[param]))]) + ".npy"
        save(data_save_path, data)


path = "flags/Original/"
raw_data = []

total_classes = os.listdir(path)

total_count = int(np.sum([len(os.listdir(path + str(i))) for i in total_classes]))

labels = []

shuffling_list = np.arange(total_count)
np.random.seed(5)
np.random.shuffle(shuffling_list)

transformation_dataset = np.ones((total_count,size[1],size[0],3))

overall_count = 0
completed_total = int(np.sum(np.array([len(dataset) for dataset in transformation_transformation_dict])))*total_count

for dataset in range(len(datasets)):
    dataset_param = datasets[dataset]

    transformation = transformation_transformation_dict[datasets[dataset]]
    space_required = len(transformation)
    if(datasets[dataset] == 'txty'):
        space_required = int(space_required*space_required)

    if(dataset_name != 'All' and dataset_name != 'raw_data'):
        if(dataset_name != dataset_param):
            continue
        completed_total = space_required*total_count
    if(dataset_name == 'raw_data'):
        completed_total = total_count

    
    for number_of_times in range(space_required):
        if(dataset_name != 'raw_data' and replace == False):
            txty_length = len(transformation_transformation_dict[dataset_param])
            if(dataset_param != 'txty' and str(param) + "_" + str(transformation_transformation_dict[dataset_param][number_of_times]) in os.listdir(path)):
                continue
            elif(dataset_param == 'txty' and str(param) + "_" + str(transformation_transformation_dict[dataset_param][int(number_of_times/txty_length)]) + "_" + str(transformation_transformation_dict[dataset_param][int(number_of_times%txty_length)]) in os.listdir(path)):
                continue

        count = 0
        for vara in range(len(total_classes)):
            classes = total_classes[vara]
        
            new_path = path + str(classes)
            images = os.listdir(new_path)
            
            for varb in range(len(images)):
                image = images[varb]
                img = cv2.imread(new_path+'/'+image)
                
                if(dataset_param != 'txty'):
                    sys.stdout.write('\r')
                    sys.stdout.write("Dataset: {}_{} for images {:.3f}% and Overall Percentage: {:.4f}%".format(str(dataset_param),str(transformation_transformation_dict[param][number_of_times]),100*(count/total_count), 100*(overall_count/completed_total)))
                    sys.stdout.flush()
                else:
                    sys.stdout.write('\r')
                    sys.stdout.write("Dataset: {}_{}_{} for images {:.3f}% and Overall Percentage: {:.4f}%".format(str(dataset_param),str(transformation_transformation_dict[param][int(number_of_times/len(transformation_transformation_dict[param]))]),str(transformation_transformation_dict[param][int(number_of_times%/len(transformation_transformation_dict[param]))]),100*(vara/len(total_classes) + (varb/len(images))/10), 100*(overall_count/completed_total)))
                    sys.stdout.flush()
            
                if img is not None:
                    if(dataset_name != 'raw_data'):
                        if(dataset_param == 'zxzy'):
                            apply_zxzy(img, transformation, count, dataset_param, number_of_times)
                        elif(dataset_param == 'txty'):
                            apply_txty(img, transformation, count, dataset_param, number_of_times)
                        elif(dataset_param == 'contrast'):
                            apply_contrast(img, transformation, count, dataset_param, number_of_times)    
                        elif(dataset_param == 'blur_avg' or datasets[dataset] == 'blur_median'):
                            apply_blur(img, transformation, count, dataset_param, number_of_times)    
                        else:
                            apply_once (img, transformation, count, dataset_param, number_of_times)

                    if(dataset == 0 and number_of_times == 0 and dataset_name == 'raw_data'):
                        img_arr = cv2.resize(img, (size[0],size[1]))
                        raw_data.append(img_arr)
                        labels.append(classes)
                    count = count + 1
                overall_count += 1
        if(dataset_name == 'raw_data'):
            raw_data=np.asarray(raw_data)
            raw_data=raw_data[shuffling_list]
            raw_data = np.asarray(raw_data)
            save('flags/Modified_DataSets1/' + "raw_data" + '.npy', raw_data)

            labels=np.array(labels)
            labels=labels[shuffling_list]
            labels = asarray(labels)
            save('flags/Modified_DataSets1/' + "labels" + '.npy', labels)
            sys.exit()

        save_dataset(dataset_param, number_of_times)
    
    if(dataset_name != 'All'):
        sys.exit()
    print("Completed Dataset:" + str(dataset/len(datasets)))
    print("\n\n\n")
   

