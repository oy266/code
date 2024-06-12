import copy
import os

import cv2
import numpy as np
from pycocotools.coco import COCO

# from ..dataloading import get_yolox_datadir

# from .datasets_wrapper import CacheDataset, cache_read_img
# from .cocovid import CocoVID

from yolox.data.dataloading import get_yolox_datadir
from yolox.data.datasets.datasets_wrapper import CacheDataset, cache_read_img
from yolox.data.datasets.cocovid import CocoVID
from yolox.data.vid_data_augment import VidTrainTransform
from torch.utils.data import DataLoader
import torch
import random

def remove_useless_info(coco):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in coco.dataset:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)


class VidCOCODataset(CacheDataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        name="train",
        img_size=(416, 416),
        preproc=None,
        cache=False,
        cache_type="ram",
        num_frames=3,
        gap=1
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        # if data_dir is None:
        #     data_dir = os.path.join(get_yolox_datadir(), "COCO")
        self.filter_key_img = True
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds() # 所有image的id号
        self.num_imgs = len(self.ids)
        self.class_ids = sorted(self.coco.getCatIds()) # 所有类别的id号
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_coco_annotations() # self.annotations[0] (array([[134., 212., 150., 228.,   0.]]), (256, 256), (512, 512), '050000.bmp')

        self.cocovid = CocoVID(os.path.join(self.json_file))
        self.num_frames = num_frames # 输入视频帧数
        self.gap = gap

        path_filename = [os.path.join(name, anno[3]) for anno in self.annotations]
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )

    def __len__(self):
        return self.num_imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False) # img id -> ann id
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )
        # (返回回归框+class 0开始)，(原图尺寸)，(输入尺寸)，(图片名)
        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        # if '.bmp' in file_name:
        #     file_name = file_name.replace('.bmp', '.jpg')


        img_file = os.path.join(self.data_dir, file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)

    def read_ref_img(self, index):
        return self.load_resized_img(index)


    def pull_item(self, index):
        id_ = self.ids[index]
        label, origin_image_size, _, _ = self.annotations[index]
        img = self.read_img(index)

        return img, copy.deepcopy(label), origin_image_size, np.array(id_)

    def get_frames_ids(self, img_ids, img_id, gap=1):
        frames_ids = []
        # if img_id - (self.num_frames) < img_ids[0]:
        #     # ref_frames_ids = [img_id] * (self.num_frames-1)
        #     offset = self.num_frames - (img_id - img_ids[0] +1)
        #     # frames_range = [img_id-img_ids[0], img_id+offset]
        #     for id in range(img_ids[0], img_id+offset+1):
        #         frames_ids.append(id)
        # else:
        #     for id in range(img_id - (self.num_frames) + 1, img_id+1):
        #         frames_ids.append(id)

        num_ref_frames = self.num_frames - 1
        for i in range(num_ref_frames):
            ref_id = img_id - (i+1) * gap
            ref_id = max(img_ids[0], ref_id)
            frames_ids.append(ref_id)

        return frames_ids

    def get_frames_ids_with_interval(self, img_ids, img_id, interval=5):
        left = max(img_ids[0], img_id - interval)
        right = min(img_ids[-1], img_id + interval)
        # sample_range = list(range(left, right))
        # frames_ids = random.sample(sample_range, self.num_frames)
        frames_ids = [left, img_id, right]
        return frames_ids

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        if isinstance(index, tuple):
            index = index[1]

        img, target, img_info, img_id = self.pull_item(index)
        # index starts form 0, while img_id starts from 1
        imgs = []
        targets = []
        imgs_ids = []
        # img_infos = [img_info] * (self.num_frames)
        img_infos = [img_info]
        # imgs.append(img)
        id_ = self.ids[index] # img id
        im_ann = self.coco.loadImgs(id_)[0]
        video_id = im_ann['video_id']

        img_ids = self.cocovid.get_img_ids_from_vid(video_id)

        # frames_ids = self.get_frames_ids_with_interval(img_ids, id_)

        frames_ids = self.get_frames_ids(img_ids, id_, gap=self.gap)
        frames_ids.sort()
        for id in frames_ids:
            index_ = id - 1
            img_ref = self.read_ref_img(index_)
            imgs.append(img_ref)
            # tar_ref = self.annotations[index_][0]
            # targets.append(tar_ref)
            # imgs_ids.append(np.array(id))

        imgs.append(img)
        targets.append(target)
        imgs_ids.append(int(img_id))

        if self.preproc is not None:
            imgs, targets = self.preproc(imgs, targets, self.img_size)
        return imgs, targets, img_infos, imgs_ids


def dataset_collate(batch): # 训练的时候只用到images和bboxes
    images = []
    bboxes = []
    img_infos_h = []
    img_infos_w = []
    img_ids = []
    for imgs, targets, img_infos, imgs_ids in batch:
        images.append(imgs)
        bboxes.append(targets)
        # img_infos_h.append(img_info[0])
        # img_infos_w.append(img_info[1])
        img_infos_h.append([img_info[0] for img_info in img_infos])
        img_infos_w.append([img_info[1] for img_info in img_infos])
        # img_id = img_id.tolist()
        imgs_ids = [id for id in imgs_ids]
        img_ids.append(imgs_ids)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    images = images.permute(0, 2, 1, 3, 4)
    # bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    for i, video_boxes in enumerate(bboxes):
        for j, img_boxes in enumerate(bboxes[i]):
            bboxes[i][j] = torch.from_numpy(img_boxes).type(torch.FloatTensor)
    # bboxes = torch.stack(bboxes)
    for i, video_boxes in enumerate(bboxes):
        bboxes[i] = torch.stack(video_boxes)
    bboxes = torch.stack(bboxes)
    bboxes = bboxes.squeeze()
    # img_infos_h = torch.tensor(img_infos_h)
    # img_infos_w = torch.tensor(img_infos_w)
    img_infos = [img_infos_h, img_infos_w]
    img_ids = torch.tensor(img_ids)
    return images, bboxes, img_infos, img_ids




if __name__ == "__main__":
    vid_dataset = VidCOCODataset(data_dir='/home/yj/Datasets/feiji',
                                 json_file='/home/yj/Datasets/feiji/daub_train.json',
                              img_size=(512, 512), preproc=VidTrainTransform(max_labels=2,
                flip_prob=1.0,
                hsv_prob=1.0), num_frames=5, gap=5)
    vid_dataset.__getitem__(6684)
    train_dataloader = DataLoader(vid_dataset, shuffle=True, batch_size=2, collate_fn=dataset_collate)
    # vid_dataset.__getitem__(0)
    for index, batch in enumerate(train_dataloader): # B, C, T, H, W = x.shape # [4, 3, 6, 224, 224]
        images, targets = batch[0], batch[1]
        print(index)
        print(images.shape)
        print(targets.shape)
        print(targets)
        print(batch[2])
        print(batch[3])
        if index == 2:
            break
    pass
