import numpy as np
import cv2
import math
from glob import glob

def calc_mean(x):
	m = np.zeros((x.shape[0], 1))

	it = np.nditer(x, flags=['multi_index'])
	while not it.finished:
		m[it.multi_index[0]] += it[0]
		it.iternext()

	m /= x.shape[1]
	return m

def train(p, n):
	image_files = glob('images/**/[134579].pgm')
	temp = image_files[6:12]
	image_files += temp
	del image_files[6:12]

	x = np.zeros((p, n))
	img_index = 0
	
	for fname in image_files:
		img = cv2.imread(fname, 0)

		x[:,img_index] = img.flatten().transpose()
		img_index += 1

	m = calc_mean(x)
	x -= m

	u, s, v = np.linalg.svd(x)

	return u, x, m

def count_votes(mins):
	votes = np.zeros(10)

	for guess in mins:
		votes[guess // 6] += 1

	return np.argmax(votes)

def classify(img, model, k, space, m):
	img -= m
	proj = np.dot(space, img)
	dif = model - np.tile(proj, (1, 60))
	dif = np.linalg.norm(dif, axis=0, keepdims=True)
	guesses = np.argmin(dif, axis=1)
	return count_votes(guesses)

def test(k, model, space, m):
	# create a list of testing image files
	image_files = glob('images/**/[2681]*.pgm')
	image_files = [item for item in image_files if item not in glob('images/**/1.pgm')]
	temp = image_files[4:8]
	image_files += temp
	del image_files[4:8]

	# Try the classify test images and return success rate
	counter = 0
	successes = 0
	for fname in image_files:
		img = np.array(cv2.imread(fname, 0).flatten(), ndmin=2, dtype=np.float).transpose()
		neighbor = classify(img, model, k, space, m)
		if counter // 4 == neighbor:
			successes += 1
		counter += 1
	return successes / counter
	
# get the result from svd, the mean adjusted training set, and the mean of the set
u, x, m = train(10304, 60)
for k in [1, 2, 3, 6, 10, 20, 30]:
	# calculate the k subspace
	space = u[:, :k].transpose()
	# project the mean adjusted training set onto the subspace
	model = np.dot(space, x)
	# test and print the succss rate
	success = test(k, model, space, m)
	print('k={}'.format(k), 'success_rate={}'.format(success))