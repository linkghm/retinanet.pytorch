import torch
from math import sqrt
from utils import box_iou, box_nms, change_box_order, meshgrid

from torch.autograd import Variable


class DataEncoder:
    def __init__(self):
        self.anchor_areas = [512*512, 256*256, 128*128, 64*64, 32*32]
        self.aspect_ratios = [1/2, 1, 2/1]
        self.scale_ratios = [1, pow(2, 1/3), pow(2, 2/3)]
        self.num_levels = len(self.anchor_areas)
        self.num_anchors = len(self.aspect_ratios) * len(self.scale_ratios)
        self.anchor_edges = self.calc_anchor_edges()

    def calc_anchor_edges(self):
        anchor_edges = []
        for area in self.anchor_areas:
            for ar in self.aspect_ratios:
                if ar < 1:
                    height = sqrt(area)
                    width = height * ar
                else:
                    width = sqrt(area)
                    height = width / ar
                for sr in self.scale_ratios:
                    anchor_height = height * sr
                    anchor_width = width * sr
                    anchor_edges.append((anchor_width, anchor_height))
        return torch.Tensor(anchor_edges).view(self.num_levels, self.num_anchors, 2)

    def get_anchor_boxes(self, input_size):
        fm_sizes = [(input_size / pow(2, i + 3)).ceil() for i in range(self.num_levels)]

        boxes = []
        for i in range(self.num_levels):
            fm_size = fm_sizes[i]
            grid_size = (input_size / fm_size).floor()
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = meshgrid(fm_w, fm_h) + 0.5  # [fm_h * fm_w, 2]
            xy = (xy * grid_size).view(fm_w, fm_h, 1, 2).expand(fm_w, fm_h, 9, 2)
            wh = self.anchor_edges[i].view(1, 1, 9, 2).expand(fm_w, fm_h, 9, 2)
            box = torch.cat([xy, wh], 3)  # [x, y, w, h]
            boxes.append(box.view(-1, 4))
        return torch.cat(boxes, 0)

    def encode(self, boxes, labels, input_size):
        if isinstance(input_size, int):
            input_size = torch.Tensor([input_size, input_size])
        else:
            input_size = torch.Tensor(input_size)

        anchor_boxes = self.get_anchor_boxes(input_size)
        boxes = change_box_order(boxes, 'xyxy2xywh')
        boxes = boxes.float()
        ious = box_iou(anchor_boxes, boxes, order='xywh')
        max_ious, max_ids = ious.max(1)
        boxes = boxes[max_ids]

        loc_xy = (boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:]
        loc_wh = torch.log(boxes[:, 2:] / anchor_boxes[:, 2:])
        loc_targets = torch.cat([loc_xy, loc_wh], 1)

        cls_targets = 1 + labels[max_ids]
        cls_targets[max_ious < 0.4] = 0
        cls_targets[(max_ious >= 0.4) & (max_ious < 0.5)] = -1
        return loc_targets, cls_targets

    def encode_protos(self, boxes, labels, input_size, desired_label=None):
        if isinstance(input_size, int):
            input_size = torch.Tensor([input_size, input_size])
        else:
            input_size = torch.Tensor(input_size)

        # ignore everything but desired label
        if desired_label>=0:
            boxes = boxes.masked_select((labels == desired_label).view(-1, 1).expand(boxes.size(0), boxes.size(1))).view((labels == desired_label).sum(), boxes.size(1))
            labels = labels.masked_select(labels == desired_label)

        anchor_boxes = self.get_anchor_boxes(input_size)
        boxes = change_box_order(boxes, 'xyxy2xywh')
        boxes = boxes.float()  # list of gt boxes
        ious = box_iou(anchor_boxes, boxes, order='xywh') #(n_anch, n_boxes) ious of boxes with anchors
        max_ious, max_ids = ious.max(1) # the max iou values and indxs of anchs and boxes per anch, so each indx refers to gt box and label indx
        boxes = boxes[max_ids]

        loc_xy = (boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:]
        loc_wh = torch.log(boxes[:, 2:] / anchor_boxes[:, 2:])
        loc_targets = torch.cat([loc_xy, loc_wh], 1) # calculate the movements needed to better align anchor to gt

        cls_targets = 1 + labels[max_ids]  # cls_targets (n_anch) containing the labels+1 of the best aligned gt box per anch box
        cls_targets[max_ious < 1] = -1  # force only one gt anchor
        # cls_targets[max_ious < .95] = -1  # force very close to 1 one anchor, only very high overlaps
        # cls_targets[max_ious < 0.4] = 0  # set other flag if overlap of anchor with GT is < 40%
        cls_targets[(max_ious >= 0.4) & (max_ious < 0.5)] = -1  # set ignore flag if overlap is between 40% and 50%
        # cls_targets[max_ious < 0.1] = 0  # set other flag if overlap of anchor with GT is < 10%
        # cls_targets[(max_ious >= 0.4) & (max_ious < 0.5)] = -1  # set ignore flag if overlap is between 40% and 50%

        if desired_label>=0:
            cls_targets[labels[max_ids] != desired_label] = -1  # set ignore flag if this isn't the label we are looking for

            if cls_targets.max() <= 0:  # no samples so need to take best iou one for desired label to prevent Nans later in pipeline
                (s_max_ious, s_mi_inds) = torch.sort(max_ious, descending=True)  # sort the max_ious
                cls_targets[s_mi_inds[0]] = desired_label+1  # assign the ind of that anchor that has highest iou

        assert (cls_targets.max() > 0)
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, input_size):
        CLS_THRESH = 0.05
        NMS_THRESH = 0.3

        if isinstance(input_size, int):
            input_size = torch.Tensor([input_size, input_size])
        else:
            input_size = torch.Tensor(input_size)

        # anchor_boxes = self.get_anchor_boxes(input_size)
        anchor_boxes = Variable(self.get_anchor_boxes(input_size).cuda())
        loc_xy = loc_preds[:, :2]
        loc_wh = loc_preds[:, 2:]
        xy = loc_xy * anchor_boxes[:, 2:] + anchor_boxes[:, :2]
        wh = loc_wh.exp() * anchor_boxes[:, 2:]
        boxes = torch.cat([xy, wh], 1)
        boxes = change_box_order(boxes, 'xywh2xyxy')

        score, labels = cls_preds.max(1)
        ids = (score > CLS_THRESH) & (labels > 0)
        ids = ids.nonzero().squeeze()
        if len(ids) > 0:
            keep = box_nms(boxes[ids], score[ids], threshold=NMS_THRESH)
            return boxes[ids][keep], labels[ids][keep]
        else:
            return None, None
