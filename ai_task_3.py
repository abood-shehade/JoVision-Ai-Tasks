import cv2
import numpy as np
import pandas as pd
import os

def load_image(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    sensor_data = image[:, width//2:]
    return sensor_data

def apply_threshold(sensor_data, lower_threshold=50, upper_threshold=200):
    gray = cv2.cvtColor(sensor_data, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, lower_threshold, upper_threshold)
    lower_bar_region = sensor_data[258:264, 0:256]

    lower_bar_hsv = cv2.cvtColor(lower_bar_region, cv2.COLOR_BGR2HSV)
    
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    pressure_flag = False
    green_mask = cv2.inRange(lower_bar_hsv, lower_green, upper_green)
    if np.sum(green_mask) / green_mask.size > 0.5:
        pressure_flag = True
    
    return mask, pressure_flag
    

def detect_fingers(mask, flag, coordinates, intensity_threshold=1900):
    finger_data = []
    
    for finger, (top_left, bottom_right) in coordinates.items():
        roi = mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        significant_pixels = roi[roi==255]
        if flag:
            print(f'finger: {finger} intensity: {len(significant_pixels)}')
        if len(significant_pixels) > intensity_threshold and flag:
            finger_data.append(1)
        else:
            finger_data.append(0)
    
    return finger_data

def save_to_excel(finger_data_list, output_file):
    df = pd.DataFrame(finger_data_list, columns=["Thumb", "Index", "Middle", "Ring", "Pinky"])
    df.to_excel(output_file, index=False)

def process_images(image_folder, output_file, coordinates):
    finger_data_list = []
    
    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        sensor_data = load_image(image_path)
        mask, flag = apply_threshold(sensor_data)
        finger_data = detect_fingers(mask,flag, coordinates)
        finger_data_list.append(finger_data)
    
    save_to_excel(finger_data_list, output_file)


coordinates = {'pinky': [(0,0),(100,23)], 'ring': [(0,48),(100,71)], 'middle' : [(0,80),(100,103)], 
                      'index' : [(0,120),(100,143)], 'thumb' : [(200,144),(231,254)]}

image_folder = r'D:\Tasks\Ai Tasks\task3Images'
output_file = "finger_pressure_data.xlsx"

process_images(image_folder, output_file, coordinates)

