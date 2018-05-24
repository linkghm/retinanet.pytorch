import os
import random
from PIL import Image
import xml.etree.ElementTree as ET

import torch
from torch.utils.data.dataset import Dataset

from voc.annotations import AnnotationDir
from voc.bbox import BoundingBox


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
    def __init__(self, image_dir,
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