from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
img_path = r"./data/train/100.jpg"


img = Image.open(img_path).convert('RGB')

plt.subplot(221)
plt.title("origin")
plt.imshow(img)

img = img.filter(ImageFilter.MedianFilter(size=1))

plt.subplot(222)
plt.title("MedianFilter(size=1)")
plt.imshow(img)

img = img.filter(ImageFilter.MedianFilter(size=3))

plt.subplot(223)
plt.title("MedianFilter(size=3)")
plt.imshow(img)

img = img.filter(ImageFilter.MedianFilter(size=5))

plt.subplot(224)
plt.title("MedianFilter(size=5)")
plt.imshow(img)
plt.show()