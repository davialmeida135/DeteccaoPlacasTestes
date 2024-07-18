import os
import shutil
import re

# Define the source directory and target directories
source_dir = "D:/Github/DeteccaoPlacas/Characters_South_America.v2i.yolov9/train/images"
target_dir_letters = "D:/Github/DeteccaoPlacas/Characters_South_America.v2i.yolov9/train/selected_images"
target_dir_others = "D:/Github/DeteccaoPlacas/Characters_South_America.v2i.yolov9/train/discarded_images"

# Create target directories if they don't exist
os.makedirs(target_dir_letters, exist_ok=True)
os.makedirs(target_dir_others, exist_ok=True)

# Regular expression to check if a filename starts with three letters
pattern = re.compile(r'^[a-zA-Z]{3}')

# Iterate over the files in the source directory
for filename in os.listdir(source_dir):
    file_path = os.path.join(source_dir, filename)
    
    # Check if it's a file (not a directory)
    if os.path.isfile(file_path):
        # Check if the filename starts with three letters
        if pattern.match(filename):
            # Move the file to the target directory for files starting with three letters
            shutil.move(file_path, os.path.join(target_dir_letters, filename))
        else:
            # Move the file to the target directory for other files
            shutil.move(file_path, os.path.join(target_dir_others, filename))