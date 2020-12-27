
from PIL import Image
import torch
from torch.utils.data import Dataset
import os

from PIL import ImageFilter
import matplotlib.pyplot as plt


TRAIN_PATH = r"./data/train/"
source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97+26)]
source += [chr(i) for i in range(65, 65+26)]
alphabet = ''.join(source)



def img_loader(img_path):
    """
    # : show image before filter and after filter.
    :param img_path: path of image
    :return: image
    """
    img = Image.open(img_path).convert('RGB')
    # plt.subplot(211)
    # plt.imshow(img)

    img = img.filter(ImageFilter.MedianFilter(size=3))
    # plt.subplot(212)
    # plt.imshow(img)
    return img





def one_hot(target_str, alphabet, num_class):
    """ One-hot coding for the text on image.
    :return: one-hot code of target_str.
    """
    code = []
    for char in target_str:
        vec = [0] * num_class
        vec[alphabet.find(char)] = 1
        code += vec
    return code


def make_dataset(data_path, alphabet, num_class, df):
    """generate a data set which is consist of the path of image and the one-hot coding text on the image.
    :param data_path: the path of image
    :param alphabet:use to code
    :param num_class: the number of character categories
    :param df: A DataFrame after after divided into test set or training set,
            consist of name of image and true label on image.
    :return:data set
    """
    img_names = os.listdir(data_path)
    dataset = []
    for img_name in img_names:
        # if image name in df
        if img_name not in df.index:
            continue
        img_path = os.path.join(data_path, img_name)
        target_str = df.loc[img_name]['label']
        code = one_hot(target_str, alphabet, num_class)
        # a map between the  one-hot coded text that on the image and the path of image.
        dataset.append((img_path, code))
    return dataset

class CaptchaData(Dataset):
    def __init__(self, data_path, num_class=62, num_char=4,
                 transform=None, target_transform=None, alphabet=alphabet, df=None):#62,4
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.num_class = num_class
        self.num_char = num_char
        self.transform = transform
        self.target_transform = target_transform
        self.alphabet = alphabet
        self.df = df
        self.samples = make_dataset(self.data_path, self.alphabet,
                                    self.num_class, self.df)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, target = self.samples[index]
        img = img_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, torch.Tensor(target)

