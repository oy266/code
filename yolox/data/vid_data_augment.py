import math
import random

import cv2
import numpy as np

from yolox.utils import myxyxy2cxcywh


def augment_hsv(imgs, hgain=5, sgain=30, vgain=30):
    for i in range(len(imgs)):
        hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
        hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
        hsv_augs = hsv_augs.astype(np.int16)
        img_hsv = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2HSV).astype(np.int16)

        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

        cv2.cvtColor(img_hsv.astype(imgs[i].dtype), cv2.COLOR_HSV2BGR, dst=imgs[i])  # no return needed


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )


def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes

    return targets


def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    if len(targets) > 0:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)

    return img, targets


def _mirror(images, boxes, prob=0.5):
    _, width, _ = images[0].shape
    if random.random() < prob:
        for i, image in enumerate(images):
            image = image[:, ::-1]
            images[i] = image
        for i, box in enumerate(boxes):
            box[:, 0::2] = width - box[:, 2::-2]
            boxes[i] = box
    return images, boxes


def preproc(imgs, input_size, swap=(2, 0, 1)):
    img = imgs[0]
    # if len(img.shape) == 3:
    #     padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    # else:
    #     padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    padded_imgs = []
    for img in imgs:
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        padded_imgs.append(padded_img)

    return padded_imgs, r


class VidTrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.0, hsv_prob=1.0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob

    def __call__(self, images, targets, input_dim):
        # boxes = targets[:, :4].copy()
        # labels = targets[:, 4].copy()
        boxes = [target[:, :4].copy() for target in targets]
        labels = [target[:, 4].copy() for target in targets]
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            images, r_o = preproc(images, input_dim)
            return images, targets

        image_o = images[0].copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = [target[:, :4].copy() for target in targets]
        labels_o = [target[:, 4].copy() for target in targets]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = myxyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(images)
        images_t1, boxes = _mirror(images, boxes, self.flip_prob)
        height, width, _ = images_t1[0].shape
        images_t2, r_ = preproc(images_t1, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = myxyxy2cxcywh(boxes) # boxes是反转后的boxes
        boxes = [box * r_ for box in boxes]

        # mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        mask_bs = [np.minimum(box[:, 2], box[:, 3]) > 1 for box in boxes]
        # boxes_t = boxes[mask_b]
        # labels_t = labels[mask_b]
        boxes_t = [box[mask_b] for box, mask_b in zip(boxes, mask_bs)]
        labels_t = [label[mask_b] for label, mask_b in zip(labels, mask_bs)]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o
        # labels对齐
        # labels_t = np.expand_dims(labels_t, 1) # array([0.]) -> array([[0.]])

        # targets_t = np.hstack((labels_t, boxes_t))

        labels_t = [np.expand_dims(label, 1) for label in labels_t]
        targets_t = [np.hstack((label, box)) for label, box in zip(labels_t, boxes_t)]
        padded_labels = [np.zeros((self.max_labels, 5))] * len(targets_t)
        # padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
        #     : self.max_labels
        # ]
        for i, (padded_label, target_t) in enumerate(zip(padded_labels, targets_t)):
            padded_label[range(len(target_t))[: self.max_labels]] = target_t[:self.max_labels]
        # padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        padded_labels = [np.ascontiguousarray(padded_label, dtype=np.float32) for padded_label in padded_labels]
        return images_t2, padded_labels


class VidValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))