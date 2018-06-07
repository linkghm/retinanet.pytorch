import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage import io, transform
import xml.etree.ElementTree as ET

import torch
from torch.utils.data.dataset import Dataset

from voc.annotations import AnnotationDir
from voc.bbox import BoundingBox

random.seed(7)

class VocLikeDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, imageset_fn, image_ext, classes, encoder, transform=None, val=False):
        self.image_dir_path = image_dir
        self.image_ext = image_ext
        with open(imageset_fn) as f:
            self.filenames = [fn.rstrip() for fn in f.readlines()]
        self.annotation_dir = AnnotationDir(annotation_dir, self.filenames, classes, '.xml', 'voc')
        self.filenames = list(self.annotation_dir.ann_dict.keys())
        self.encoder = encoder
        self.transform = transform
        self.val = val

    def __getitem__(self, index):
        fn = self.filenames[index]
        image_fn = '{}{}'.format(fn, self.image_ext)
        image_path = os.path.join(self.image_dir_path, image_fn)
        image = Image.open(image_path)
        boxes = self.annotation_dir.get_boxes(fn)
        example = {'image': image, 'boxes': boxes}
        if self.transform:
            example = self.transform(example)
        return example

    def __len__(self):
        return len(self.filenames)

    def collate_fn(self, batch):
        imgs = [example['image'] for example in batch]
        boxes  = [example['boxes'] for example in batch]
        labels = [example['labels'] for example in batch]
        img_sizes = [img.size()[1:] for img in imgs]

        max_h = max([im.size(1) for im in imgs])
        max_w = max([im.size(2) for im in imgs])
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, max_h, max_w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            im = imgs[i]
            imh, imw = im.size(1), im.size(2)
            inputs[i,:,:imh,:imw] = im

            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(max_w, max_h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        if not self.val:
            return inputs, torch.stack(loc_targets), torch.stack(cls_targets)
        return inputs, img_sizes, torch.stack(loc_targets), torch.stack(cls_targets)



class VocLikeProtosDataset(Dataset):
    def __init__(self,
                 image_dir,
                 annotation_dir,
                 imageset_fn,
                 image_ext,
                 classes,
                 n_way,
                 n_support,
                 n_query,
                 encoder,
                 transform=None,
                 val=False):
        self.image_dir_path = image_dir
        self.image_ext = image_ext
        self.classes = classes
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query

        with open(imageset_fn) as f:
            self.filenames = [fn.rstrip() for fn in f.readlines()]
        self.annotations, self.classes_samples = self.build_annotations(annotation_dir, ".xml")
        self.filenames = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform
        self.val = val

    def __getitem__(self, index):
        if index is int:
            fn = self.filenames[index]
        else:
            fn = index
        image_fn = '{}{}'.format(fn, self.image_ext)
        image_path = os.path.join(self.image_dir_path, image_fn)
        image = Image.open(image_path)
        boxes = self.annotations[fn]
        example = {'image': image, 'boxes': boxes}
        if self.transform:
            example = self.transform(example)
        return example


    # def __getitem__(self, episode_index):
    #     fn = self.episode_ids[episode_index]
    #     image_fn = '{}{}'.format(fn, self.image_ext)
    #     image_path = os.path.join(self.image_dir_path, image_fn)
    #     image = Image.open(image_path)
    #     boxes = self.annotations[fn]
    #     example = {'image': image, 'boxes': boxes, 'desired_label': self.desired_classes[episode_index]}
    #     if self.transform:
    #         example = self.transform(example)
    #     return example

    def __len__(self):
        return len(self.filenames)

    # def generate_episode(self):
    #     classes = random.sample(self.classes, self.n_way)
    #
    #     self.episode_ids = [random.sample(self.classes_samples[self.classes.index(cls)], self.n_support + self.n_query) for cls in classes]
    #     self.episode_ids = [item for sublist in self.episode_ids for item in sublist]
    #     self.desired_classes = [[self.classes.index(cls)]*(self.n_support + self.n_query) for cls in classes]
    #     self.desired_classes = [item for sublist in self.desired_classes for item in sublist]


    def load_episode(self):
        classes = random.sample(self.classes, self.n_way)

        episode_ids = []
        desired_classes = []
        episode_ids_tmp = [random.sample(self.classes_samples[self.classes.index(cls)], self.n_support + self.n_query) for cls in classes]
        desired_classes_tmp = [[self.classes.index(cls)] * (self.n_support + self.n_query) for cls in classes]

        # order to be [c1s1,c1s2,..,c10s5,c1q1,c1q2,...,c10q20] so can pass all support first so can mean asap
        for ci in range(self.n_way):
            episode_ids += episode_ids_tmp[ci][:self.n_support]
            desired_classes += desired_classes_tmp[ci][:self.n_support]
        for ci in range(self.n_way):
            episode_ids += episode_ids_tmp[ci][self.n_support:]
            desired_classes += desired_classes_tmp[ci][self.n_support:]

        episode = [self.__getitem__(id) for id in episode_ids]

        imgs = [example['image'] for example in episode]
        boxes = [example['boxes'] for example in episode]
        labels = [example['labels'] for example in episode]
        img_sizes = [img.size()[1:] for img in imgs]

        max_h = max([im.size(1) for im in imgs])
        max_w = max([im.size(2) for im in imgs])
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, max_h, max_w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            im = imgs[i]
            imh, imw = im.size(1), im.size(2)
            inputs[i, :, :imh, :imw] = im

            loc_target, cls_target = self.encoder.encode_protos(boxes[i], labels[i], input_size=(max_w, max_h),
                                                         desired_label=desired_classes[i])
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        if not self.val:
            return inputs, torch.stack(loc_targets), torch.stack(cls_targets)
        return inputs, img_sizes, torch.stack(loc_targets), torch.stack(cls_targets)


    def collate_fn(self, batch):
        imgs = [example['image'] for example in batch]
        boxes  = [example['boxes'] for example in batch]
        labels = [example['labels'] for example in batch]
        img_sizes = [img.size()[1:] for img in imgs]

        max_h = max([im.size(1) for im in imgs])
        max_w = max([im.size(2) for im in imgs])
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, max_h, max_w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            im = imgs[i]
            imh, imw = im.size(1), im.size(2)
            inputs[i,:,:imh,:imw] = im

            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(max_w, max_h), desired_label=8)
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        if not self.val:
            return inputs, torch.stack(loc_targets), torch.stack(cls_targets)
        return inputs, img_sizes, torch.stack(loc_targets), torch.stack(cls_targets)

    def build_annotations(self, annotation_dir, ext):
        box_dict = {}
        classes_samples = {}
        for fn in self.filenames:
            boxes = []
            tree = ET.parse(os.path.join(annotation_dir, fn + ext))
            ann_tag = tree.getroot()

            size_tag = ann_tag.find('size')
            image_width = int(size_tag.find('width').text)
            image_height = int(size_tag.find('height').text)

            for obj_tag in ann_tag.findall('object'):
                label = obj_tag.find('name').text

                box_tag = obj_tag.find('bndbox')
                left = int(box_tag.find('xmin').text)
                top = int(box_tag.find('ymin').text)
                right = int(box_tag.find('xmax').text)
                bottom = int(box_tag.find('ymax').text)

                box = BoundingBox(left, top, right, bottom, image_width, image_height, self.classes.index(label))
                if self.classes.index(label) not in classes_samples:
                    classes_samples[self.classes.index(label)] = set([os.path.splitext(fn)[0]])
                else:
                    classes_samples[self.classes.index(label)].add(os.path.splitext(fn)[0])
                boxes.append(box)
            if len(boxes) > 0:
                box_dict[os.path.splitext(fn)[0]] = boxes
            else:
                self.filenames.remove(fn)
        return box_dict, classes_samples


class OmniglotDetectDataset(Dataset):
    def __init__(self,
                 base_dir,
                 n_objects_p_i,
                 n_classes_p_i,
                 encoder,
                 n_way,
                 batch_size=None,
                 n_support=None,
                 n_query=None,
                 split="train",
                 n_classes=100,
                 transform=None,
                 val=False,
                 classification=False):
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.batch_size = batch_size
        self.split = split

        self.n_classes_p_i = n_classes_p_i
        self.n_objects_p_i = n_objects_p_i
        self.img_size = (150, 150)
        self.char_scales = (.5, 3)
        self.force_square = True

        self.imgs = self.build_set(base_dir)
        self.classes = list(range(min(len(self.imgs), n_classes)))

        self.encoder = encoder
        self.transform = transform
        self.val = val

        self.classification = classification

    def __getitem__(self, index):
        image, boxes = self.generate_sample(index)
        image = Image.fromarray((image*255).astype(dtype=np.uint8))
        example = {'image': image, 'boxes': boxes}
        if self.transform:
            example = self.transform(example)
        return example

    def __len__(self):
        return len(self.imgs)


    def load_protos_episode(self):
        classes = random.sample(self.classes, self.n_way)

        indexs = []
        # desired_classes = []
        episode_ids_tmp = [random.sample(list(range(len(self.imgs[self.classes.index(cls)]))), self.n_support + self.n_query) for cls in classes]
        desired_classes_tmp = [[self.classes.index(cls)] * (self.n_support + self.n_query) for cls in classes]

        # order to be [c1s1,c1s2,..,c10s5,c1q1,c1q2,...,c10q20] so can pass all support first so can mean asap
        for ci in range(self.n_way):
            indexs+= list(map(list, zip(*[desired_classes_tmp[ci][:self.n_support], episode_ids_tmp[ci][:self.n_support]])))
            # indexs.append([episode_ids_tmp[ci][:self.n_support], desired_classes_tmp[ci][:self.n_support]])
        for ci in range(self.n_way):
            indexs += list(map(list, zip(*[desired_classes_tmp[ci][self.n_support:], episode_ids_tmp[ci][self.n_support:]])))
            # indexs.append([episode_ids_tmp[ci][self.n_support:], desired_classes_tmp[ci][self.n_support:]])

        episode = [self.__getitem__(id) for id in indexs]

        imgs = [example['image'] for example in episode]
        boxes = [example['boxes'] for example in episode]
        labels = [example['labels'] for example in episode]
        img_sizes = [img.size()[1:] for img in imgs]

        max_h = max([im.size(1) for im in imgs])
        max_w = max([im.size(2) for im in imgs])
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, max_h, max_w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            im = imgs[i]
            imh, imw = im.size(1), im.size(2)
            inputs[i, :, :imh, :imw] = im

            loc_target, cls_target = self.encoder.encode_protos(boxes[i], labels[i], input_size=(max_w, max_h),
                                                         desired_label=indexs[i][0])
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        if not self.val:
            return inputs, torch.stack(loc_targets), torch.stack(cls_targets)
        return inputs, img_sizes, torch.stack(loc_targets), torch.stack(cls_targets)



    def load_mem_episode(self, e, view=False):
        loc_targetss = []
        cls_targetss = []
        cls_dict = []  # 0 can't exist here as we reserve that for background
        inputss = []
        img_sizess = []
        avail_classes = set(self.classes)
        for bi in range(self.batch_size):
            assert len(avail_classes) > self.n_way  # make sure we have enough classes avail to chose n_way from
            # this is in place of the orig paper and code which can take same class for different batch breaking the unseen
            # property of few shot, but its okay as num classes is high compared to nway so random choice of same class is rare
            # but here instead we assert that the classes must be different

            classes = random.sample(list(avail_classes), self.n_way)

            for cls in classes:
                avail_classes.remove(cls)

            indexs = []
            # desired_classes = []
            episode_ids_tmp = [random.sample(list(range(len(self.imgs[self.classes.index(cls)]))), self.n_support) for cls in classes]
            desired_classes_tmp = [[self.classes.index(cls)] * self.n_support for cls in classes]

            # order to be [c1s1,c3s1,c2s1,..,c4s5,c1s5,c3s5] so get a sample from all classes before moving onto next support set
            for si in range(self.n_support):
                for ci in sorted(list(range(self.n_way)), key=lambda k: random.random()):
                    indexs.append([desired_classes_tmp[ci][si], episode_ids_tmp[ci][si]])

            episode = [self.__getitem__(id) for id in indexs]

            imgs = [example['image'] for example in episode]
            boxes = [example['boxes'] for example in episode]
            labels = [example['labels'] for example in episode]
            img_sizes = [img.size()[1:] for img in imgs]

            max_h = max([im.size(1) for im in imgs])
            max_w = max([im.size(2) for im in imgs])
            num_imgs = len(imgs)
            inputs = torch.zeros(num_imgs, 3, max_h, max_w)

            loc_targets = []
            cls_targets = []
            for i in range(num_imgs):
                im = imgs[i]
                imh, imw = im.size(1), im.size(2)
                inputs[i, :, :imh, :imw] = im

                # labels are 1 --> n_way+1
                # loc_target, cls_target = self.encoder.encode_protos(boxes[i],
                #                                                     torch.LongTensor([classes.index(int(labels[i])) + (bi*self.n_way)+1]),
                #                                                     input_size=(max_w, max_h),
                #                                                     desired_label=classes.index(indexs[i][0]) + (bi*self.n_way)+1)

                # # labels are their original class label
                loc_target, cls_target = self.encoder.encode_protos(boxes[i],
                                                                    labels[i],
                                                                    input_size=(max_w, max_h),
                                                                    desired_label=indexs[i][0])
                loc_targets.append(loc_target)
                cls_targets.append(cls_target)

            inputss.append(inputs)
            img_sizess.append(img_sizes)
            loc_targetss.append(torch.stack(loc_targets))
            cls_targetss.append(torch.stack(cls_targets))

        # Stack and transpose so they are shape [eplen, batchsize, c, w, h]
        inputss = torch.stack(inputss).transpose(0,1)
        loc_targetss = torch.stack(loc_targetss).transpose(0,1)
        cls_targetss = torch.stack(cls_targetss).transpose(0,1)
        # img_sizess = torch.stack(img_sizess).transpose(0,1)

        if view:
            # from skimage.viewer import ImageViewer

            from PIL import Image
            for i in range(self.n_way*self.n_support):
                imgs = inputss.numpy()[i]
                imgs = np.swapaxes(imgs, 0,1)
                imgs = imgs.reshape((imgs.shape[0], imgs.shape[1]*imgs.shape[2], imgs.shape[3]))
                imgs = np.swapaxes(imgs,0,2)

                img = Image.fromarray((imgs * 255).astype(np.uint8))
                img.save("/media/hayden/Storage21/MODELS/PROTINANET/ins/e"+str(e)+"_"+str(i)+".png")

                # viewer = ImageViewer(imgs)
                # viewer.show()

        # if not self.val:
        return inputss, loc_targetss, cls_targetss

        # return inputss, img_sizess, loc_targetss, cls_targetss


    def build_set(self, dir):
        from skimage import io, transform
        imgs = []
        with open(os.path.join(dir, "SPLITS/vinyals", self.split + ".txt"), 'r') as f:
            lines = f.readlines()
        lines = [line.split("/") for line in lines]

        for si in tqdm(range(len(lines))):
            alphabet = lines[si][0]
            character = lines[si][1]
            rot = int(lines[si][2][3:])

            image_dir = os.path.join(dir, 'IMAGES', alphabet, character)

            characters = []
            for filename in os.listdir(image_dir):
                filename = os.path.join(image_dir, filename)
                img = io.imread(filename)
                characters.append(transform.rotate(img, rot))

            imgs.append(characters)

        print("Images Loaded.")
        return imgs

    def apply_noise(self, image, noise_typ='speckle_norm'):
        if len(image.shape) < 3:
            image = image.reshape(image.shape[0], image.shape[1], 1)
        row, col, ch = image.shape

        if noise_typ == "gauss":
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = image + gauss
            return noisy.squeeze()
        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy.squeeze()
        elif noise_typ == "speckle":
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = image + image * gauss
            return noisy.squeeze()
        elif noise_typ == "speckle_norm":
            gauss = np.random.rand(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = image * gauss
            return noisy.squeeze()

    def crop_char(self, img, pad=2):
        itemindex = np.where(img == 0)
        min_x = max(min(itemindex[1]) - pad, 0)
        max_x = min(max(itemindex[1]) + pad, img.shape[1])
        min_y = max(min(itemindex[0]) - pad, 0)
        max_y = min(max(itemindex[0]) + pad, img.shape[0])
        middle_x = int((min_x + max_x) / 2)
        middle_y = int((min_y + max_y) / 2)
        w = max_x - min_x
        h = max_y - min_y
        size = max(w, h)
        if self.force_square:
            l = middle_x - int(size / 2)
            r = middle_x + int(size / 2)
            t = middle_y - int(size / 2)
            b = middle_y + int(size / 2)
        else:
            l = middle_x - int(w / 2)
            r = middle_x + int(w / 2)
            t = middle_y - int(h / 2)
            b = middle_y + int(h / 2)

        imgt = img[t:b, l:r]
        if imgt.shape[0] > 10 and imgt.shape[1] > 10: # ensures that we actually get an image out
            img = imgt
        return img

    def get_rand_pos(self, img_size, char_size, precovered=[set([]), set([])]):
        # coords of tl placement
        possible_x = set(range(img_size[1]))
        possible_y = set(range(img_size[0]))

        # knock out image edges so char doesn't hover off img edge
        possible_x -= (set(range(img_size[1] - char_size[1], img_size[1])))  # knock out L/R sides
        possible_y -= (set(range(img_size[0] - char_size[0], img_size[0])))  # knock out T/B sides

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
            if not set(range(xt, xt + char_size[1])) & precovered[1]:
                # if xt not in precovered[0] and xt + char_size[0] not in precovered[0]:
                x = xt
                break
        for yt in possible_y:
            if not set(range(yt, yt + char_size[0])) & precovered[0]:
                # if yt not in precovered[1] and yt + char_size[1] not in precovered[1]:
                y = yt
                break

        if not x or not y:
            return None, None, None, None, None

        padding = 2
        coverage = [set(range(y - padding, y + char_size[1] + padding)),
                    set(range(x - padding, x + char_size[0] + padding))]

        return y, x, char_size[0], char_size[1], coverage

    def generate_sample(self, index):
        char_ind = index[0]
        samp_ind = index[1]

        back = np.ones(self.img_size)  # *255

        # apply first character we are interested in
        img = self.imgs[char_ind][samp_ind]
        if self.classification:
            back = img
            boxes = [BoundingBox(1, 1, 1+(img.shape[0]-2), 1+(img.shape[1]-2), img.shape[0], img.shape[0], self.classes.index(char_ind))]
        else:
            n_characters = len(self.classes)

            img = self.crop_char(img)

            for i in range(10000000):

                scl = random.uniform(self.char_scales[0], self.char_scales[1])
                imgb = Image.fromarray((img * 255).astype(np.uint8))
                imgb = imgb.resize((int(scl*img.shape[0]), int(scl*img.shape[1])), Image.ANTIALIAS)
                imgb = np.array(imgb)/255
                # imgb = transform.rescale(img, scl, mode='constant', cval=1.0)
                # imgb = img
                y, x, h, w, cover = self.get_rand_pos(self.img_size, imgb.shape)
                if x:
                    break

            assert x

            back[y:y + h, x:x + w] = imgb
            boxes = [BoundingBox(x, y, x+w, y+h, self.img_size[1], self.img_size[0], self.classes.index(char_ind))]

            r_clss = random.sample(range(n_characters), (self.n_classes_p_i - 1))
            r_clss.append(char_ind)

            for r_obj_ind in range(self.n_objects_p_i - 1):
                r_cls = random.choice(r_clss)
                r_smp = random.randint(0, len(self.imgs[r_cls]) - 1)

                img = self.imgs[r_cls][r_smp]
                img = self.crop_char(img)

                for i in range(100):
                    scl = random.uniform(self.char_scales[0], self.char_scales[1])
                    imgb = Image.fromarray((img * 255).astype(np.uint8))
                    imgb = imgb.resize((int(scl * img.shape[0]), int(scl * img.shape[1])), Image.ANTIALIAS)
                    imgb = np.array(imgb) / 255
                    # imgb = transform.rescale(img, scl, mode='constant', cval=1.0)
                    # imgb = img
                    y, x, h, w, c = self.get_rand_pos(self.img_size, imgb.shape, precovered=cover)
                    if x:
                        cover[0] = cover[0] | c[0]
                        cover[1] = cover[1] | c[1]
                        back[y:y + h, x:x + w] = imgb
                        boxes.append(BoundingBox(x, y, x+w, y+h, self.img_size[1], self.img_size[0], self.classes.index(r_cls)))
                        break

        # back = self.apply_noise(back)
        back = np.expand_dims(back, axis=2)
        back = np.repeat(back, 3, axis=2)
        # back = self.apply_noise(back)
        # for box in boxes:
        #     back = print_bb(back, box)

        return back, boxes