import os
from os import path as osp
import mmcv
import numpy as np
from concurrent import futures as futures

class S3DISData(object):
    """S3DIS dataset used to generate infos for semantic segmentation task.

    Args:
        data_root (str): Root path of the raw data.
        ann_file (str): The generated scannet infos.
        split (str, optional): Set split type of the data. Default: 'train'.
        num_points (int, optional): Number of points in each data input.
            Default: 8192.
        label_weight_func (function, optional): Function to compute the
            label weight. Default: None.
    """

    def __init__(self,
                 data_root,
                 ann_file,
                 split='Area_1',
                 num_points=4096,
                 label_weight_func=None):
        self.data_root = data_root
        self.data_infos = mmcv.load(ann_file)
        self.split = split
        self.num_points = num_points

        self.all_ids = np.arange(13)  # all possible ids
        self.cat_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                 12])  # used for seg task
        self.ignore_index = len(self.cat_ids)

        self.cat_id2class = np.ones((self.all_ids.shape[0],), dtype=np.int) * \
            self.ignore_index
        for i, cat_id in enumerate(self.cat_ids):
            self.cat_id2class[cat_id] = i

        # label weighting function is taken from
        # https://github.com/charlesq34/pointnet2/blob/master/scannet/scannet_dataset.py#L24
        self.label_weight_func = (lambda x: 1.0 / np.log(1.2 + x)) if \
            label_weight_func is None else label_weight_func

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int, optional): Number of threads to be used.
                Default: 4.
            has_label (bool, optional): Whether the data has label.
                Default: True.
            sample_id_list (list[int], optional): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            info = dict()
            pc_info = {
                'num_features': 6,
                'lidar_idx': f'{self.split}_{sample_idx}'
            }
            info['point_cloud'] = pc_info
            pts_filename = osp.join(self.root_dir, 's3dis_data',
                                    f'{self.split}_{sample_idx}_point.npy')
            pts_instance_mask_path = osp.join(
                self.root_dir, 's3dis_data',
                f'{self.split}_{sample_idx}_ins_label.npy')
            pts_semantic_mask_path = osp.join(
                self.root_dir, 's3dis_data',
                f'{self.split}_{sample_idx}_sem_label.npy')

            points = np.load(pts_filename).astype(np.float32)
            pts_instance_mask = np.load(pts_instance_mask_path).astype(np.int)
            pts_semantic_mask = np.load(pts_semantic_mask_path).astype(np.int)

            mmcv.mkdir_or_exist(osp.join(self.root_dir, 'points'))
            mmcv.mkdir_or_exist(osp.join(self.root_dir, 'instance_mask'))
            mmcv.mkdir_or_exist(osp.join(self.root_dir, 'semantic_mask'))

            points.tofile(
                osp.join(self.root_dir, 'points',
                         f'{self.split}_{sample_idx}.bin'))
            pts_instance_mask.tofile(
                osp.join(self.root_dir, 'instance_mask',
                         f'{self.split}_{sample_idx}.bin'))
            pts_semantic_mask.tofile(
                osp.join(self.root_dir, 'semantic_mask',
                         f'{self.split}_{sample_idx}.bin'))

            info['pts_path'] = osp.join('points',
                                        f'{self.split}_{sample_idx}.bin')
            info['pts_instance_mask_path'] = osp.join(
                'instance_mask', f'{self.split}_{sample_idx}.bin')
            info['pts_semantic_mask_path'] = osp.join(
                'semantic_mask', f'{self.split}_{sample_idx}.bin')
            info['annos'] = self.get_bboxes(points, pts_instance_mask,
                                            pts_semantic_mask)

            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def get_seg_infos(self):
        scene_idxs, label_weight = self.get_scene_idxs_and_label_weight()
        save_folder = osp.join(self.data_root, 'seg_info')
        mmcv.mkdir_or_exist(save_folder)
        np.save(
            osp.join(save_folder, f'{self.split}_resampled_scene_idxs.npy'),
            scene_idxs)
        np.save(
            osp.join(save_folder, f'{self.split}_label_weight.npy'),
            label_weight)
        print(f'{self.split} resampled scene index and label weight saved')

    def _convert_to_label(self, mask):
        """Convert class_id in loaded segmentation mask to label."""
        if isinstance(mask, str):
            if mask.endswith('npy'):
                mask = np.load(mask)
            else:
                mask = np.fromfile(mask, dtype=np.int64)
        label = self.cat_id2class[mask]
        return label

    def get_scene_idxs_and_label_weight(self):
        """Compute scene_idxs for data sampling and label weight for loss
        calculation.

        We sample more times for scenes with more points. Label_weight is
        inversely proportional to number of class points.
        """
        num_classes = len(self.cat_ids)
        num_point_all = []
        label_weight = np.zeros((num_classes + 1, ))  # ignore_index
        for data_info in self.data_infos:
            label = self._convert_to_label(
                osp.join(self.data_root, data_info['pts_semantic_mask_path']))
            num_point_all.append(label.shape[0])
            class_count, _ = np.histogram(label, range(num_classes + 2))
            label_weight += class_count

        # repeat scene_idx for num_scene_point // num_sample_point times
        sample_prob = np.array(num_point_all) / float(np.sum(num_point_all))
        num_iter = int(np.sum(num_point_all) / float(self.num_points))
        scene_idxs = []
        for idx in range(len(self.data_infos)):
            scene_idxs.extend([idx] * int(round(sample_prob[idx] * num_iter)))
        scene_idxs = np.array(scene_idxs).astype(np.int32)

        # calculate label weight, adopted from PointNet++
        label_weight = label_weight[:-1].astype(np.float32)
        label_weight = label_weight / label_weight.sum()
        label_weight = self.label_weight_func(label_weight).astype(np.float32)

        return scene_idxs, label_weight

def create_indoor_info_file(data_path,
                            pkl_prefix='sunrgbd',
                            save_path=None,
                            use_v1=False,
                            workers=4):
    """Create indoor information file.

    Get information of the raw data and save it to the pkl file.

    Args:
        data_path (str): Path of the data.
        pkl_prefix (str, optional): Prefix of the pkl to be saved.
            Default: 'sunrgbd'.
        save_path (str, optional): Path of the pkl to be saved. Default: None.
        use_v1 (bool, optional): Whether to use v1. Default: False.
        workers (int, optional): Number of threads to be used. Default: 4.
    """
    assert os.path.exists(data_path)
    assert pkl_prefix in ['sunrgbd', 'scannet', 's3dis'], \
        f'unsupported indoor dataset {pkl_prefix}'
    save_path = data_path if save_path is None else save_path
    assert os.path.exists(save_path)


    if pkl_prefix == 's3dis':
        # S3DIS doesn't have a fixed train-val split
        # it has 6 areas instead, so we generate info file for each of them
        # in training, we will use dataset to wrap different areas
        splits = [f'Area_{i}' for i in [1, 2, 3, 4, 5, 6]]
        for split in splits:
            dataset = S3DISData(
                data_root=data_path,
                ann_file=filename,
                split=split,
                num_points=4096,
                label_weight_func=lambda x: 1.0 / np.log(1.2 + x))
            info = dataset.get_infos(num_workers=workers, has_label=True)
            filename = os.path.join(save_path,
                                    f'{pkl_prefix}_infos_{split}.pkl')
            mmcv.dump(info, filename, 'pkl')
            print(f'{pkl_prefix} info {split} file is saved to {filename}')
            dataset = S3DISData(
                data_root=data_path,
                ann_file=filename,
                split=split,
                num_points=4096,
                label_weight_func=lambda x: 1.0 / np.log(1.2 + x))
            dataset.get_seg_infos()

def s3dis_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for s3dis dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)

if "__name__"=="__main__":
    # root_path: the path save sourcedata; out_dir: the path output 
    root_path = ""
    out_dir = ""
    s3dis_data_prep(
                root_path=root_path,
                info_prefix="s3dis",
                out_dir=out_dir,
                workers=4)