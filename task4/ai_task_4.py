import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
from pathlib import Path
import shutil

image_folder = Path(r'D:\Tasks\Ai Tasks\task4\Chess_dataset')
output_folder = Path(r'D:\Tasks\Ai Tasks\task4\results')



for image_path in image_folder.glob('*.jpg'):  
    results = model(image_path)

    label_file = output_folder / (image_path.stem + '.txt')
    results.save_txt(save_dir=output_folder)
    annotated_img = output_folder / (image_path.stem + '_pred.jpg')
    shutil.copy(image_path, annotated_img)
    results.show(save=True, save_dir=str(output_folder))
