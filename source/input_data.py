import os
import numpy as np
import json
import pandas as pd
import cv2

from tensorflow import keras
from keras.applications.mobilenet import preprocess_input
from keras.utils import to_categorical

RAW_DP_DIR = '/data/doodle/train_sampled_raw'
DP_DIR = '/data/doodle/train_sampled'
CT_DIR = '/data/doodle/train'

BASE_SIZE = 256
BASE_SIZE_RAW = 6000
NCATS = 340
IN_CHANNEL = 3
MAXPXL = 255
MINPXL = 0
N_FEATURES = 6

def extract_all_features(raw_strokes, size=256, lw=6, time_color=True, lastdrop_r = 0.0):
	stx_min, sty_min = 99999, 99999
	stx_max, sty_max = 0,0
	ett = 0  # How fast to complete less than 20 seconds

	for t, stroke in enumerate(raw_strokes):
		if t == len(raw_strokes) -1:
			ett = int(stroke[2][-1])
		for i in range(len(stroke[0])):
			stx_min = min(stx_min, int(stroke[0][i]))
			stx_max = max(stx_max, int(stroke[0][i]))
			sty_min = min(sty_min, int(stroke[1][i]))
			sty_max = max(sty_max, int(stroke[1][i]))

	limit_ett = 20*1000   
	ofs = 15

	if int(sty_max-sty_min+2*ofs) > 6000 or int(stx_max-stx_min+2*ofs)  > 6000:
		img1 = np.zeros((6000,6000,IN_CHANNEL), np.uint8)
		img2 = np.zeros((6000,6000,IN_CHANNEL), np.uint8)
	else:
		img1 = np.zeros((int(sty_max-sty_min+2*ofs), int(stx_max-stx_min+2*ofs),IN_CHANNEL), np.uint8)
		img2 = np.zeros((int(sty_max-sty_min+2*ofs), int(stx_max-stx_min+2*ofs),IN_CHANNEL), np.uint8)

	(maxrow, maxcol, _) = img1.shape
	oldc0 = 0
	for t, stroke in enumerate(raw_strokes):	 
		inertia_x = 0
		inertia_y = 0
		pre_st_t = 0 
		maxdraw_t = stroke[2][-1] - stroke[2][0]
		#c2 = int(((dt-st+ofs)/(maxdraw_t+ofs))*255)
		n_points = len(stroke[0])
		for i in range(len(stroke[0]) - 1):
			sx = int(stroke[0][i]) - stx_min + ofs
			sy = int(stroke[1][i]) - sty_min + ofs
			st = stroke[2][i]
			dx = int(stroke[0][i+1]) - stx_min + ofs
			dy = int(stroke[1][i+1]) - sty_min + ofs
			dt = stroke[2][i+1]

			time = abs(dt-st)
			if time == 0:
				time = 1
			#print(dt, st, time, np.sqrt(time), np.sqrt((sx-dx)*(sx-dx) + (sy-dy)*(sy-dy)))
			c0 = min(int((np.sqrt((sx-dx)*(sx-dx) + (sy-dy)*(sy-dy)) / np.sqrt(time))*255.0), 255)
			c1 = min(int((abs(c0 - oldc0)/np.sqrt(time))*255.0), 255)
			c2 = min(int((np.sqrt((inertia_x-dx)*(inertia_x-dx) + (inertia_y-dy)*(inertia_y-dy)) \
					/ np.sqrt(time*time))*255.0), 255)
			c3 = min(n_points, 255)
			c4 = min(int(np.sqrt((sx-dx)*(sx-dx) + (sy-dy)*(sy-dy))), 255)
			c5 = min(int((sx-dx)/(sy-dy+0.001)), 255)
			color1 = (c0, c1, c2)
			color2 = (c3, c4, c5)
			_ = cv2.line(img1, (sx, sy), (dx, dy), color1, lw)
			_ = cv2.line(img2, (sx, sy), (dx, dy), color2, lw)

			inertia_x = 2*dx - sx
			inertia_y = 2*dy - sy
			oldc0 = c0
	
	img1 = cv2.resize(img1, (size, size))
	img2 = cv2.resize(img2, (size, size))
	img = np.concatenate([img1, img2], axis=-1)

	return img

def image_extractor_xd_raw(size, batchsize, ks, lw=6, time_color=True):
	while True:
		for k in np.random.permutation(ks):
			filename = os.path.join(RAW_DP_DIR, 'train_k{}.csv.gz'.format(k))
			for df in pd.read_csv(filename, chunksize=batchsize):
				df['drawing'] = df['drawing'].apply(json.loads)
				x = np.zeros((len(df), size, size, N_FEATURES))
				for i, raw_strokes in enumerate(df.drawing.values):
					x[i, :, :, :] = extract_all_features(raw_strokes, size=size, lw=6)
				x = (x.astype(np.float32)-MINPXL)/(MAXPXL-MINPXL)
				y = keras.utils.to_categorical(df.y, num_classes=NCATS)
				yield x, y

def f2cat(filename: str) -> str:
	return filename.split('.')[0]

def list_all_categories(ct_dir):
	files = os.listdir(ct_dir)
	return sorted([f2cat(f) for f in files], key=str.lower)

def draw_cv2(raw_strokes, size=256, lw=6, augmentation=False):
	img = np.zeros((BASE_SIZE, BASE_SIZE, IN_CHANNEL), np.uint8)
	stroke_cnt = len(raw_strokes)
	for t, stroke in enumerate(raw_strokes):
		n_points = len(stroke[0])
		for i in range(len(stroke[0]) - 1):
			sx, dx = stroke[0][i:i+2]
			sy, dy = stroke[1][i:i+2]

			c0 = 255 - min(t, 10)*13
			c1 = min(n_points, 255)
			c2 = int(min(len(stroke[0]), 255) * (c0/255))
			#cl = (255, 255-abs(sx-dx), 255-abs(sy-dy))
			cl = (c0, c1, c2)
			_ = cv2.line(img, (sx, sy), (dx, dy), cl, lw)
	if size != BASE_SIZE:
		img = cv2.resize(img, (size, size))
	if augmentation:
		if random.random() > 0.5:
			img = np.fliplr(img)
	return img

def image_generator_xd(size, batchsize, ks, lw=6, time_color=True):
	while True:
		for k in np.random.permutation(ks):
			filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))
			for df in pd.read_csv(filename, chunksize=batchsize):
				df['drawing'] = df['drawing'].apply(json.loads)
				x = np.zeros((len(df), size, size, IN_CHANNEL))
				for i, raw_strokes in enumerate(df.drawing.values):
					x[i, :, :, :] = draw_cv2(raw_strokes, size=size, lw=lw)
				x = (x.astype(np.float32)-MINPXL)/(MAXPXL-MINPXL)
				y = keras.utils.to_categorical(df.y, num_classes=NCATS)
				yield x, y
				
def df_to_image_array_xd(df, size, lw=6, time_color=True):
	df['drawing'] = df['drawing'].apply(json.loads)
	x = np.zeros((len(df), size, size, IN_CHANNEL))
	for i, raw_strokes in enumerate(df.drawing.values):
		x[i, :, :, :] = draw_cv2(raw_strokes, size=size, lw=lw)
	x = (x.astype(np.float32)-MINPXL)/(MAXPXL-MINPXL)
	return x


##############################################################
def draw_cv2_raw(raw_strokes, size=256, lw=6, time_color=True, lastdrop_r = 0.0):
	stx_min, sty_min = 99999, 99999
	stx_max, sty_max = 0,0
	ett=0  # How fast to complete less than 20 seconds

	for t, stroke in enumerate(raw_strokes):
		if t == len(raw_strokes) -1:
			ett = int(stroke[2][-1])
		for i in range(len(stroke[0])):
			stx_min = min(stx_min, int(stroke[0][i]))
			stx_max = max(stx_max, int(stroke[0][i]))
			sty_min = min(sty_min, int(stroke[1][i]))
			sty_max = max(sty_max, int(stroke[1][i]))

	limit_ett = 20*1000
	ofs = 15

	if int(sty_max-sty_min+2*ofs) > 6000 or int(stx_max-stx_min+2*ofs)  > 6000:
		img = np.zeros((6000,6000,3), np.uint8)
	else:
		img = np.zeros((int(sty_max-sty_min+2*ofs), int(stx_max-stx_min+2*ofs),3), np.uint8)

	for t, stroke in enumerate(raw_strokes):
		n_points = len(stroke[0])
		for i in range(len(stroke[0]) - 1):
			sx = int(stroke[0][i]) - stx_min +ofs
			sy = int(stroke[1][i]) - sty_min +ofs
			st = stroke[2][i]
			dx = int(stroke[0][i + 1])- stx_min +ofs
			dy = int(stroke[1][i + 1])- sty_min +ofs
			et = stroke[2][i+1]
			time = abs(et - st)
			time = 1 if time == 0 else time

			c0 = 255 - min(t, 10)*13
			c1 = min(n_points, 255)/np.sqrt(time)
			c2 = min(int((np.sqrt((sx-dx)*(sx-dx) + (sy-dy)*(sy-dy)) / time)*255.0), 255)
			_ = cv2.line(img, (sx, sy), (dx, dy), (c0,c1,c2), lw)

			'''
			if i==0:
				color_inter = int((float(et-pre_st_t)/limit_ett)*245)+10
				_ = cv2.circle(img, (sx, sy), lw, (0,0,color_inter), -1) ##interval time

			if i==len(stroke[0])-2 and t == len(raw_strokes) -1:
				color_end = int((float(ett)/(limit_ett)*245))+10
				_ = cv2.circle(img, (sx, sy), lw, (0,color_end,0), -1) ##end time

			inertia_x = 2*dx -sx
			inertia_y = 2*dy-sy
			pre_st_t=et
			'''

	return cv2.resize(img, (size, size))
##############################################################

def image_generator_xd_raw(size, batchsize, ks, lw=6, time_color=True):
	while True:
		for k in np.random.permutation(ks):
			filename = os.path.join(RAW_DP_DIR, 'train_k{}.csv.gz'.format(k))
			for df in pd.read_csv(filename, chunksize=batchsize):
				df['drawing'] = df['drawing'].apply(json.loads)
				x = np.zeros((len(df), size, size, IN_CHANNEL))
				for i, raw_strokes in enumerate(df.drawing.values):
					x[i, :, :, :] = draw_cv2_raw(raw_strokes, size=size, lw=lw)
				x = (x.astype(np.float32)-MINPXL)/(MAXPXL-MINPXL)
				y = keras.utils.to_categorical(df.y, num_classes=NCATS)
				yield x, y
				
def df_to_image_array_xd_raw(df, size, lw=6, time_color=True):
	df['drawing'] = df['drawing'].apply(json.loads)
	x = np.zeros((len(df), size, size, IN_CHANNEL))
	for i, raw_strokes in enumerate(df.drawing.values):
		x[i, :, :, :] = draw_cv2_raw(raw_strokes, size=size, lw=lw)
	x = (x.astype(np.float32)-MINPXL)/(MAXPXL-MINPXL)
	return x

