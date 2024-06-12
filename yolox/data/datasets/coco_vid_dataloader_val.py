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
from yolox.data.vid_data_augment import VidTrainTransform, VidValTransform
from torch.utils.data import DataLoader
import torch
import math
import copy
import numpy as np
import random

def ChooseFrame(List, Gap, num_frames):
    ret = []
    start_id = 0
    max_gap = Gap*num_frames
    num = len(List) // max_gap # num是video clip的数量
    for i in range(num):
        start_id = i * max_gap
        for j in range(Gap):
            tmp = []
            for k in range(num_frames):
                tmp.append(List[start_id + j + k * Gap])
            ret.append(copy.deepcopy(tmp)) # ret[[77, 230, 281], [80, 120, 369], [195, 288, 355], [39, 343, 302], ...]

    if num * max_gap == len(List):
        return ret

    new_list = List[num * max_gap:]
    random.shuffle(new_list)
    ret.extend(np.array_split(new_list, len(new_list) // num_frames))
    return ret


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


class VidCOCODatasetVal(CacheDataset):
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
        num_frames = 3,
        is_shuffle=False,
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

        self.cocovid = CocoVID(os.path.join(self.data_dir, "annotations", self.json_file))
        self.num_frames = num_frames #video clip frames num

        self.video_clips = []
        self.video_ids = self.cocovid.get_vid_ids()


        for vid_id in self.video_ids:
            single_video_img_ids = self.cocovid.get_img_ids_from_vid(vid_id)
            while len(single_video_img_ids) < num_frames:
                single_video_img_ids.extend(copy.deepcopy(single_video_img_ids))
            nums = math.ceil(len(single_video_img_ids)* 1.0 / num_frames) # 4
            offset = nums * num_frames - len(single_video_img_ids) # 1
            if offset != 0 :
                single_video_img_ids.extend(copy.deepcopy(single_video_img_ids[-offset:])) # 不整除就补全
            if is_shuffle:
                random.shuffle(single_video_img_ids)
            self.video_clips.extend(ChooseFrame(single_video_img_ids, gap, num_frames))



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
        return len(self.video_clips)

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

    def load_resized_img_with_path(self, path_img):
        file_name = os.path.join(self.data_dir, path_img)
        if '.bmp' in file_name:
            file_name = file_name.replace('.bmp', '.jpg')
        img = cv2.imread(file_name)
        assert img is not None, f"file named {img} not found"
        return img

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

    def get_ref_frames_ids(self, img_ids, img_id):
        if img_id - self.num_frames < img_ids[0]:
            ref_frames_ids = [img_id] * self.num_frames
        else:
            ref_frames_ids = []
            for id in range(img_id - self.num_frames, img_id):
                ref_frames_ids.append(id)

        return ref_frames_ids

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index = index[1]
        video_clip = self.video_clips[index]

        imgs = []
        targets = []
        img_ids = []
        img_infos = []

        for ids, i in enumerate(video_clip):
            coco = self.coco
            img_id = self.ids[i-1]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            target = coco.loadAnns(ann_ids)
            img_info = coco.loadImgs(img_id)[0]
            path = img_info['file_name']
            img = self.load_resized_img_with_path(path)
            label, origin_image_size, _, _ = self.annotations[i-1]
            if ids == len(video_clip) -1 :
                img_infos.append(origin_image_size)
                img_ids.append(i)
            imgs.append(img)

        if self.preproc is not None:
            imgs, targets = self.preproc(imgs, targets, self.img_size)

        return imgs, targets, img_infos, img_ids




def dataset_collate_val(batch): # 训练的时候只用到images和bboxes
    images = []
    bboxes = []
    img_infos_h = []
    img_infos_w = []
    img_ids = []
    for imgs, _, img_infos, imgs_ids in batch:
        images.append(imgs)
        img_infos_h.append([img_info[0] for img_info in img_infos])
        img_infos_w.append([img_info[1] for img_info in img_infos])
        img_ids.append(imgs_ids)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    images = images.permute(0, 2, 1, 3, 4)
    img_infos_h = torch.tensor(img_infos_h)
    img_infos_w = torch.tensor(img_infos_w)
    img_infos = torch.stack((img_infos_h, img_infos_w), dim=0) # (长宽x2， bs， num_frames)
    img_ids = torch.tensor(img_ids)
    return images, bboxes, img_infos, img_ids




if __name__ == "__main__":
    vid_dataset = VidCOCODatasetVal(data_dir='/root/autodl-tmp/daub', json_file='/root/autodl-tmp/daub_val.json',
                              img_size=(512, 512), preproc=VidValTransform(legacy=False), num_frames=5)
    # vid_dataset.__getitem__(0)
    val_dataloader = DataLoader(vid_dataset, shuffle=True, batch_size=2, collate_fn=dataset_collate_val)
    # for index, batch in enumerate(train_dataloader):
    #     images, targets = batch[0], batch[1]
    #     print(index)
    #     print(images.shape)
    #     print(targets.shape)
    #     print(batch[2])
    #     print(batch[3])
    #     if index == 2:
    #         break
    # pass
    for iter, (imgs, _, info_imgs, ids) in enumerate(val_dataloader):
        print(iter)
        print(imgs.shape)
        print(info_imgs)
        print(ids)
        for (img_h, img_w, img_id) in zip(info_imgs[0], info_imgs[1], ids):
            pass # img_h, img_w 是一个batch元素里的imgs(video clip)里所有images的h和w， img_id是video_clip里所有imgs的id
