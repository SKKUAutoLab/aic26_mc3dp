from __future__ import division, print_function, absolute_import

import re
import glob
import os.path as osp
import warnings

from ..dataset import ImageDataset


class MarketAICity(ImageDataset):
    """AICity 2025 in Market1501 format.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_
    
    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [-1]
    dataset_dir = 'market_aicity'
    masks_base_dir = 'masks'
    # dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'
    cam_num = 480
    train_dir = 'train'
    query_dir = 'query'
    gallery_dir = 'test'

    masks_dirs = {
        # dir_name: (parts_num, masks_stack_size, contains_background_mask)
        'pifpaf': (36, False, '.jpg.confidence_fields.npy'),
        'pifpaf_maskrcnn_filtering': (36, False, '.jpg.npy'),
    }

    @staticmethod
    def get_masks_config(masks_dir):
        if masks_dir not in MarketAICity.masks_dirs:
            return None
        else:
            return MarketAICity.masks_dirs[masks_dir]

    def __init__(self, root='', market1501_500k=False, masks_dir=None, **kwargs):
        self.kp_dir = kwargs['config'].model.kpr.keypoints.kp_dir
        self.masks_dir = masks_dir
        if self.masks_dir in self.masks_dirs:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = self.masks_dirs[self.masks_dir]
        else:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = None, None, None
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        # self.download_dataset(self.dataset_dir, self.dataset_url)
        self.masks_dir = masks_dir


        self.train_dir = osp.join(self.dataset_dir, self.train_dir)
        self.query_dir = osp.join(self.dataset_dir, self.query_dir)
        self.gallery_dir = osp.join(self.dataset_dir, self.gallery_dir)

        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]        
        required_files = [
            self.dataset_dir, self.train_dir
        ]

        self.check_before_run(required_files)
        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)
        super(MarketAICity, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c([-\d]+)s')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            # print(pid, camid)
            if pid == -1:
                continue # junk images are just ignored
            assert 0 <= pid <= 1501 # pid == 0 means background
            assert 0 <= camid <= 480
            # camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            masks_path = self.infer_masks_path(img_path, self.masks_dir, self.masks_suffix)
            kp_path = self.infer_kp_path(img_path)
            data.append({'img_path': img_path,
                         'pid': pid,
                         'masks_path': masks_path,
                         'camid': camid,
                         'kp_path': kp_path,
                         })
        return data
