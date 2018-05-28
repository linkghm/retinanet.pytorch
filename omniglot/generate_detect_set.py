"""
Used to generate a version of the omniglot detection set

"""
import os
import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage import data
from skimage.viewer import ImageViewer
from skimage import io, transform
from PIL import Image

def load_images(split='train'):
    imgs = []
    with open(os.path.join(DATA_DIR, "SPLITS/vinyals", split+".txt"), 'r') as f:
        lines = f.readlines()
    lines = [line.split("/") for line in lines]

    alphabets = []
    for si in tqdm(range(len(lines))):
        alphabet = lines[si][0]
        character = lines[si][1]
        rot = int(lines[si][2][3:])

        image_dir = os.path.join(DATA_DIR, 'IMAGES', alphabet, character)

        characters = []
        for filename in os.listdir(image_dir):
            filename = os.path.join(image_dir, filename)
            img = io.imread(filename)
            characters.append(transform.rotate(img, rot))
        #
        # if si < 1:
        #     prev_alphabet = alphabet
        # elif prev_alphabet != alphabet:
        #     imgs.append(alphabets)
        #     alphabets = []
        #     prev_alphabet = alphabet

        imgs.append(characters)

    return imgs
# io.imsave('local_logo.png', logo)


# def one_char_per_img():

def apply_noise(noise_typ,image):
    if len(image.shape)<3:
        image = image.reshape(image.shape[0],image.shape[1],1)
    row, col, ch = image.shape

    if noise_typ == "gauss":
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy.squeeze()
    # elif noise_typ == "s&p": # errors
    #     s_vs_p = 0.5
    #     amount = 0.004
    #     out = np.copy(image)
    #     # Salt mode
    #     num_salt = np.ceil(amount * image.size * s_vs_p)
    #     coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    #     out[coords] = 1
    #
    #     # Pepper mode
    #     num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    #     coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    #     out[coords] = 0
    #     return out.squeeze()
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy.squeeze()
    elif noise_typ =="speckle":
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return noisy.squeeze()
    elif noise_typ =="speckle_norm":
        gauss = np.random.rand(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image * gauss
        return noisy.squeeze()

def crop_char(img, pad=2):
    itemindex = np.where(img == 0)
    min_x = max(min(itemindex[1])-pad, 0)
    max_x = min(max(itemindex[1])+pad, img.shape[0])
    min_y = max(min(itemindex[0])-pad, 0)
    max_y = min(max(itemindex[0])+pad, img.shape[1])
    middle_x = int((min_x+max_x)/2)
    middle_y = int((min_y+max_y)/2)
    w = max_x - min_x
    h = max_y - min_y
    size = max(w, h)
    l = middle_x - int(size/2)
    r = middle_x + int(size/2)
    t = middle_y - int(size/2)
    b = middle_y + int(size/2)

    img = img[t:b, l:r]
    return img


def get_rand_pos(img_size, char_size, precovered=[set([]), set([])]):
    # coords of tl placement
    possible_x = set(range(img_size[0]))
    possible_y = set(range(img_size[1]))

    # knock out image edges so char doesn't hover off img edge
    possible_x -= (set(range(img_size[0]-char_size[0], img_size[0])))  # knock out L/R sides
    possible_y -= (set(range(img_size[1]-char_size[1], img_size[1])))  # knock out T/B sides

    # assert we can still place
    # assert possible_x
    # assert possible_y
    if not possible_x or not possible_y:
        return None, None, None, None, None

    possible_x = list(possible_x)
    random.shuffle(possible_x)

    possible_y = list(possible_y)
    random.shuffle(possible_y)

    # get the random positions and knock out other chars placements coverages
    x = None
    y = None
    for xt in possible_x:
        if not set(range(xt, xt+char_size[0])) & precovered[0]:
        # if xt not in precovered[0] and xt + char_size[0] not in precovered[0]:
            x = xt
            break
    for yt in possible_y:
        if not set(range(yt, yt+char_size[1])) & precovered[1]:
        # if yt not in precovered[1] and yt + char_size[1] not in precovered[1]:
            y = yt
            break

    # assert we can still place
    # assert x
    # assert y
    if not x or not y:
        return None, None, None, None, None


    padding = 2
    coverage = [set(range(x-padding, x+char_size[0]+padding)),
                set(range(y-padding, y+char_size[1]+padding))]

    return x, y, char_size[0], char_size[1], coverage

def print_bb(img, bounding_box, color=np.array([0, 1, 0], dtype=np.uint8)):

    img[bounding_box[1], bounding_box[0]:bounding_box[0] + bounding_box[2]] = color
    img[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]] = color

    img[bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]] = color
    img[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0] + bounding_box[2]] = color

    return img


DATA_DIR = "/media/hayden/Storage21/DATASETS/IMAGE/OMNIGLOT/"

MERGE_ALPHA = None
CLASSES_PER_IMAGE = 1
NOISE = 'speckle_norm'
IMG_SIZES = [(500, 500)]

CHAR_SCALES = (.5,3)
CHAR_ROT = [0, 90, 180, 270]

OBJ_PER_IMG = 5

CLS_PER_IMG = 3
# assert CLS_PER_IMG <= OBJ_PER_IMG # TODO Check if this is necesary

chars = load_images('val')
n_characters = len(chars)

for char_ind in tqdm(range(n_characters)):
    for sample_ind in range(len(chars[char_ind])):

        time.sleep(2)
        size = IMG_SIZES[0]
        back = np.ones(size)#*255

        # apply first character we are interested in
        img = chars[char_ind][sample_ind]

        img = crop_char(img)
        while True:
            scl = random.uniform(CHAR_SCALES[0],CHAR_SCALES[1])
            imgb = transform.rescale(img, scl, mode='reflect')
            x,y,w,h,cover = get_rand_pos(size, imgb.shape)
            if x:
                break

        back[x:x+w,y:y+h] = imgb
        boxes = [(y, x, w, h)]

        r_clss = random.sample(range(n_characters), (CLS_PER_IMG-1))
        r_clss.append(char_ind)

        for r_obj_ind in range(OBJ_PER_IMG-1):
            r_cls = random.choice(r_clss)
            r_smp = random.randint(0, len(chars[r_cls])-1)

            img = chars[r_cls][r_smp]
            img = crop_char(img)
            count = 0
            while True:
                scl = random.uniform(CHAR_SCALES[0], CHAR_SCALES[1])
                imgb = transform.rescale(img, scl, mode='reflect')
                x, y, w, h, c = get_rand_pos(size, imgb.shape, precovered=cover)
                if x or count>100:
                    break
            cover[0] = cover[0] | c[0]
            cover[1] = cover[1] | c[1]
            back[x:x + w, y:y + h] = imgb
            boxes.append((y, x, w, h))

        # back = apply_noise(NOISE, back)
        back = np.expand_dims(back, axis=2)
        back = np.repeat(back, 3, axis=2)
        for box in boxes:
            back = print_bb(back, box)
        # viewer = ImageViewer(back)
        # viewer.show()
        # imgplot = plt.imshow(back)
    # break