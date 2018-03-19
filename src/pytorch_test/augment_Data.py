from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter
import numpy as np
import os
from tqdm import tqdm

data_path = "../../processed_data/expert_model/train/class1/"
images = next(os.walk(data_path))[2]

def flip_image(img_path):
	orig_image = Image.open(img_path).convert('RGB')
	if np.random.randint(0, 2, size=1):img = ImageOps.mirror(orig_image)
	else : img = ImageOps.flip(orig_image)
	img = img.filter(ImageFilter.GaussianBlur(radius=2))
	return img

for image in tqdm(images):
	src = data_path + image.split(".")[0] + '.jpg'
	dstn = data_path + image.split(".")[0] + '-aug.jpg'
	img = flip_image(src)
	img.save(dstn)
	 
