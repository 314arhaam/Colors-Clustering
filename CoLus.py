from numpy import array, size, fromstring, uint8
from PIL import Image, ImageDraw
from scipy.misc import imresize
from sklearn.cluster import KMeans
from math import sin, cos, pi
import matplotlib.pyplot as mpl

def centerToBB(X, Y, r):
	BB = []
	for (x,y) in zip(X,Y):
		BB += [[x+r, y+r, x-r, y-r]]
	return BB

def clustering(name, n_clusters = 5, encoding = 'HSV', performance_show = False, shape = '|', final_show = True, threshold = 1e6):
	image = Image.open(name + '.jpg').convert(encoding)
	original_image = image
	h, w = image.height, image.width
	image = array(image)
	radius = min(h,w)//4
	if shape == 'o': 	# Cirlce
		x, y = [w//2 + radius*cos(2*i*pi/n_clusters) for i in range(n_clusters)], [h//2 + radius*sin(2*i*pi/n_clusters) for i in range(n_clusters)]
	elif shape == '|': 	# Column
		x, y = [w // 2] * n_clusters, [i * h // (1 + n_clusters) for i in range(1, 1 + n_clusters)]
	if h * w >= threshold:
		scale = (h * w)/1e6
		image = array(imresize(image, 1/scale))
	L, W = size(image)//3, 3
	imageData = image.reshape([L, W])
	model = KMeans(n_clusters=n_clusters, n_jobs=1, random_state=1, verbose=True, max_iter=2)
	res = model.fit(imageData)
	colors = res.cluster_centers_/255
	if performance_show:
		performance = mpl.imshow(res.labels_.reshape(h,w))
		mpl.show()
	colours = [(int(a),int(b),int(c)) for a, b, c in res.cluster_centers_]
	boundingBox = centerToBB(x, y, radius//3)	#idependent radius
	draw = ImageDraw.Draw(original_image)
	for (box, colour) in zip(boundingBox, colours):
		draw.rectangle(box, fill=colour)
	del draw
	if final_show:
		original_image.show()
	original_image.convert('RGB').save('_colus_'+encoding+name+'.png')

if __name__ == '__main__':
	print("Initializing")
	clustering('bmw', encoding='RGB')
