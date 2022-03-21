import cv2 as cv
import numpy as num
from skimage import io
from sklearn.cluster import KMeans


def make_histogram(cluster):
    
    numLabels = num.arange(0, len(num.unique(cluster.labels_)) + 1)
    histo, _ = num.histogram(cluster.labels_, bins=numLabels)
    histo = histo.astype('float32')
    histo /= histo.sum()
    return histo


def make_bar(height, width, color):
   
    bar = num.zeros((height, width, 3), num.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    hsv_bar = cv.cvtColor(bar, cv.COLOR_BGR2HSV)
    hue, sat, val = hsv_bar[0][0]
    return bar, (red, green, blue), (hue, sat, val)


def sort_hsvs(hsv_list):
   
    bars_with_indexes = []
    for index, hsv_val in enumerate(hsv_list):
        bars_with_indexes.append((index, hsv_val[0], hsv_val[1], hsv_val[2]))
    bars_with_indexes.sort(key=lambda elem: (elem[1], elem[2], elem[3]))
    return [item[0] for item in bars_with_indexes]



url = 'http://xxx.xxx.xx.xx/capture'
img = io.imread(url)
height, width, _ = num.shape(img)


image = img.reshape((height * width, 3))


num_clusters = 5
clusters = KMeans(n_clusters=num_clusters)
clusters.fit(image)


histogram = make_histogram(clusters)

combined = zip(histogram, clusters.cluster_centers_)
combined = sorted(combined, key=lambda x: x[0], reverse=True)

bars = []
hsv_values = []
for index, rows in enumerate(combined):
    bar, rgb, hsv = make_bar(100, 100, rows[1])
    print(f'Bar {index + 1}')
    print(f'  RGB values: {rgb}')
    print(f'  HSV values: {hsv}')
    hsv_values.append(hsv)
    bars.append(bar)

sorted_bar_indexes = sort_hsvs(hsv_values)
sorted_bars = [bars[idx] for idx in sorted_bar_indexes]

cv.imshow('Sorted by HSV values', num.hstack(sorted_bars))
cv.imshow(f'{num_clusters} Most Common Colors', num.hstack(bars))
cv.waitKey(0)
