import os, pickle
from os.path import join
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_points_kernels import knn
from .build import DATASETS


class ConfigSemanticKITTI:
    k_n = 16  # KNN
    num_layers = 4  # Number of layers
    num_points = 4096 * 11  # Number of input points
    num_classes = 19  # Number of valid classes
    sub_grid_size = 0.06  # preprocess_parameter

    batch_size = 6  # batch_size during training
    val_batch_size = 20  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256]  # feature dimension
    num_sub_points = [num_points // 4, num_points // 16, num_points // 64, num_points // 256]

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None


class DataProcessing:
    @staticmethod
    def get_file_list(dataset_path, seq_list):
        # seq_list = np.sort(os.listdir(dataset_path))
        file_list = []
        for seq_id in seq_list:
            seq_path = join(dataset_path, seq_id)
            pc_path = join(seq_path, 'velodyne')
            file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])  
        file_list = np.concatenate(file_list, axis=0)
        return file_list

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """
        # offset_support = torch.tensor([support_pts.size(1)*(i+1) for i in range(support_pts.size(0))], 
        #                         device=support_pts.device, dtype=torch.int32)
        # offset_query = torch.tensor([query_pts.size(1)*(i+1) for i in range(query_pts.size(0))], 
        #                         device=query_pts.device, dtype=torch.int32)
        # support_pts = support_pts.contiguous().view(-1, support_pts.size(-1))
        # query_pts = query_pts.contiguous().view(-1, query_pts.size(-1))
        # neighbor_idx, _ = pointops.knnquery(k, support_pts, query_pts, offset_support, offset_query)
        # neighbor_idx = neighbor_idx.contiguous().view(offset_query.size(0), -1, neighbor_idx.size(-1)).long()
        # return neighbor_idx
        if not torch.is_tensor(support_pts):
            support_pts = torch.tensor(support_pts)
        if not torch.is_tensor(query_pts):
            query_pts = torch.tensor(query_pts)
        neighbor_idx, neighbor_dist2 = knn(support_pts, query_pts, k)
        return neighbor_idx

    @staticmethod
    def shuffle_idx(x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list

    @staticmethod
    def IoU_from_confusions(confusions):
        """
        Computes IoU from confusion matrices.
        :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes
        :return: ([..., n_c] np.float32) IoU score
        """

        # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
        # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)

        # Compute IoU
        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

        # Compute mIoU with only the actual classes
        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
        IoU += mask * mIoU
        return IoU

    @staticmethod
    def get_class_weights(dataset_name):
        # pre-calculate the number of points in each category
        num_per_class = []
        if dataset_name is 'S3DIS':
            num_per_class = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                                      650464, 791496, 88727, 1284130, 229758, 2272837], dtype=np.int32)
        elif dataset_name is 'Semantic3D':
            num_per_class = np.array([5181602, 5012952, 6830086, 1311528, 10476365, 946982, 334860, 269353],
                                     dtype=np.int32)
        elif dataset_name is 'SemanticKITTI':
            num_per_class = np.array([55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                                      240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                                      9833174, 129609852, 4506626, 1168181])
        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        return np.expand_dims(ce_label_weight, axis=0)



@DATASETS.register_module()
class SemanticKITTI(Dataset):
    """voxelize data for randlanet"""
    classes = [
        "unlabeled",
        "car",
        "bicycle",
        "motorcycle",
        "truck",
        "other-vehicle",
        "person",
        "bicyclist",
        "motorcyclist",
        "road",
        "parking",
        "sidewalk",
        "other-ground",
        "building",
        "fence",
        "vegetation",
        "trunk",
        "terrain",
        "pole",
        "traffic-sign",
        ]
    num_classes = 19
    num_per_class = np.array([55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                            240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                            9833174, 129609852, 4506626, 1168181])

    def __init__(self, split, data_list=None, transform=None, loop=1, **kwargs):
        self.name = 'SemanticKITTI'
        self.dataset_path = 'data/SemanticKITTI/sequences_0.06'
        self.ignored_labels = np.sort([0])
        self.split = split
        self.transform = transform
        self.loop = loop
        if data_list is None:
            if split == 'train':
                seq_list = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
            elif split == 'val':
                seq_list = ['08']
            self.data_list = DataProcessing.get_file_list(self.dataset_path, seq_list)
        else:
            self.data_list = data_list
        self.data_list = sorted(self.data_list)

    def __len__(self):
        return len(self.data_list) * self.loop

    def __getitem__(self, item):
        item = item % len(self.data_list)
        selected_pc, selected_labels, selected_idx, cloud_ind = self.spatially_regular_gen(item, self.data_list)
        data = {'pos': selected_pc, 'y': selected_labels}
        if self.transform is not None:
            data = self.transform(data)
        selected_pc = data['pos']
        return selected_pc, selected_labels, selected_idx, cloud_ind

    def spatially_regular_gen(self, item, data_list):
        # Generator loop
        cloud_ind = item
        pc_path = data_list[cloud_ind]
        pc, tree, labels = self.get_data(pc_path)
        # crop a small point cloud
        pick_idx = np.random.choice(len(pc), 1)
        selected_pc, selected_labels, selected_idx = self.crop_pc(pc, labels, tree, pick_idx)
        return selected_pc, selected_labels, selected_idx, np.array([cloud_ind], dtype=np.int32)

    def get_data(self, file_path):
        file_path = file_path.split('/')
        seq_id = file_path[-3]
        frame_id = file_path[-1].split('.')[0]
        kd_tree_path = join(self.dataset_path, seq_id, 'KDTree', frame_id + '.pkl')
        # read pkl with search tree
        with open(kd_tree_path, 'rb') as f:
            search_tree = pickle.load(f)
        points = np.array(search_tree.data, copy=False)
        # load labels
        label_path = join(self.dataset_path, seq_id, 'labels', frame_id + '.npy')
        labels = np.squeeze(np.load(label_path))
        return points, search_tree, labels

    @staticmethod
    def crop_pc(points, labels, search_tree, pick_idx):
        # crop a fixed size point cloud for training
        center_point = points[pick_idx, :].reshape(1, -1)
        select_idx = search_tree.query(center_point, k=ConfigSemanticKITTI.num_points)[1][0]
        select_idx = DataProcessing.shuffle_idx(select_idx)
        select_points = points[select_idx]
        select_labels = labels[select_idx]
        return select_points, select_labels, select_idx

    def tf_map(self, batch_pc, batch_label, batch_pc_idx, batch_cloud_idx):
        features = batch_pc
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []
        for i in range(ConfigSemanticKITTI.num_layers):
            neighbour_idx = DataProcessing.knn_search(batch_pc, batch_pc, ConfigSemanticKITTI.k_n)
            sub_points = batch_pc[:, :batch_pc.shape[1] // ConfigSemanticKITTI.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :batch_pc.shape[1] // ConfigSemanticKITTI.sub_sampling_ratio[i], :]
            up_i = DataProcessing.knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_pc = sub_points
        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [features, batch_label, batch_pc_idx, batch_cloud_idx]
        return input_list

    def collate_fn(self, batch):
        selected_pc, selected_labels, inputs = [], [], {}
        for i in range(len(batch)):
            selected_pc.append(batch[i][0])
            selected_labels.append(batch[i][1])
        selected_pc = np.stack(selected_pc)
        selected_labels = np.stack(selected_labels)
        inputs['pos'] = torch.from_numpy(selected_pc).float()
        inputs['y'] = torch.from_numpy(selected_labels).long()
        return inputs

        # selected_pc, selected_labels, selected_idx, cloud_ind = [], [], [], []
        # for i in range(len(batch)):
        #     selected_pc.append(batch[i][0])
        #     selected_labels.append(batch[i][1])
        #     selected_idx.append(batch[i][2])
        #     cloud_ind.append(batch[i][3])
        # selected_pc = np.stack(selected_pc)
        # selected_labels = np.stack(selected_labels)
        # selected_idx = np.stack(selected_idx)
        # cloud_ind = np.stack(cloud_ind)
        # flat_inputs = self.tf_map(selected_pc, selected_labels, selected_idx, cloud_ind)

        # num_layers = ConfigSemanticKITTI.num_layers
        # inputs = {}
        # inputs['xyz'] = []
        # for tmp in flat_inputs[:num_layers]:
        #     inputs['xyz'].append(torch.from_numpy(tmp).float())
        # inputs['neigh_idx'] = []
        # for tmp in flat_inputs[num_layers: 2 * num_layers]:
        #     inputs['neigh_idx'].append(tmp.long())
        # inputs['sub_idx'] = []
        # for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
        #     inputs['sub_idx'].append(tmp.long())
        # inputs['interp_idx'] = []
        # for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
        #     inputs['interp_idx'].append(tmp.long())
        # inputs['pos'] = torch.from_numpy(flat_inputs[4 * num_layers]).float()
        # inputs['y'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
        # return inputs


@DATASETS.register_module()
class SemanticKITTIDataset(Dataset):
    """raw data for spvcnn"""
    classes = [
        "unlabeled"
        "car"
        "bicycle"
        "motorcycle"
        "truck"
        "other-vehicle"
        "person"
        "bicyclist"
        "motorcyclist"
        "road"
        "parking"
        "sidewalk"
        "other-ground"
        "building"
        "fence"
        "vegetation"
        "trunk"
        "terrain"
        "pole"
        "traffic-sign"
        ]
    num_classes = 19
    num_per_class = np.array([62313931,338088, 610061, 2903453, 3652481, 607823, 204074, 82829, 275246377, 19673273, 
                        194109334, 6919208, 249415088, 113277856, 508113690, 10725351, 140272196, 4860599, 1230565]) # voxel_size 0.05  
    # num_per_class = np.array([55169650, 319507, 539395, 2564890, 3256435, 550348, 183255, 78896, 239229369, 17119551,
    #                     168788593, 6326922, 227988770, 100539516, 470824521, 9782179, 128156652, 4452709, 1163785]) # voxel_size 0.06    
    class2color = {"unlabeled": [0, 0, 0],
                    "car": [245, 150, 100],
                    "bicycle": [245, 230, 100],
                    "motorcycle": [150, 60, 30], 
                    "truck": [180, 30, 80], 
                    "other-vehicle": [255, 0, 0],
                    "person": [30, 30, 255],
                    "bicyclist": [200, 40, 255],
                    "motorcyclist": [90, 30, 150],
                    "road": [255, 0, 255],
                    "parking": [255, 150, 255],
                    "sidewalk": [75, 0, 75],
                    "other-ground": [75, 0, 175],
                    "building": [0, 200, 255],
                    "fence": [50, 120, 255],
                    "vegetation": [0, 175, 0],
                    "trunk": [0, 60, 135],
                    "terrain": [80, 240, 150],
                    "pole": [150, 240, 255],
                    "traffic-sign": [0, 0, 255]}
    cmap = [*class2color.values()]
    
    def __init__(
        self,
        split="train",
        data_root="data/SemanticKITTI",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=0,
        sources=False,
        **kwargs,
    ):
        super(SemanticKITTIDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.sources = sources

        self.data_list = self.get_data_list()
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)

    def get_data_list(self):
        split2seq = dict(
            train=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
            val=[8],
            test=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        )
        if isinstance(self.split, str):
            seq_list = split2seq[self.split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError

        data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, "sequences", seq)
            seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
            data_list += [
                os.path.join(seq_folder, "velodyne", file) for file in seq_files
            ]
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                segment = np.vectorize(self.learning_map.__getitem__)(
                    segment & 0xFFFF
                ).astype(np.int32)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)
        data_dict = dict(pos=coord, strength=strength, y=segment, name=self.get_data_name(idx))
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"
        return data_name

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0 : ignore_index,     # "unlabeled"
            1 : ignore_index,     # "outlier" mapped to "unlabeled" --------------------------mapped
            10: 1,     # "car"
            11: 2,     # "bicycle"
            13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
            15: 3,     # "motorcycle"
            16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
            18: 4,     # "truck"
            20: 5,     # "other-vehicle"
            30: 6,     # "person"
            31: 7,     # "bicyclist"
            32: 8,    # "motorcyclist"
            40: 9,     # "road"
            44: 10,    # "parking"
            48: 11,    # "sidewalk"
            49: 12,    # "other-ground"
            50: 13,    # "building"
            51: 14,   # "fence"
            52: ignore_index,     # "other-structure" mapped to "unlabeled" ------------------mapped
            60: 9,     # "lane-marking" to "road" ---------------------------------mapped
            70: 15,    # "vegetation"
            71: 16,    # "trunk"
            72: 17,    # "terrain"
            80: 18,    # "pole"
            81: 19,    # "traffic-sign"
            99: ignore_index,     # "other-object" to "unlabeled" ----------------------------mapped
            252: 1,    # "moving-car" to "car" ------------------------------------mapped
            253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
            254: 6,    # "moving-person" to "person" ------------------------------mapped
            255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
            256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
            257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
            258: 4,    # "moving-truck" to "truck" --------------------------------mapped
            259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
        }
        return learning_map

    @staticmethod
    def get_learning_map_inv(ignore_index):
        learning_map_inv = {
            ignore_index: ignore_index,  # "unlabeled", and others ignored
            1: 10,     # "car"
            2: 11,     # "bicycle"
            3: 15,     # "motorcycle"
            4: 18,     # "truck"
            5: 20,     # "other-vehicle"
            6: 30,     # "person"
            7: 31,     # "bicyclist"
            8: 32,     # "motorcyclist"
            9: 40,     # "road"
            10: 44,    # "parking"
            11: 48,    # "sidewalk"
            12: 49,    # "other-ground"
            13: 50,    # "building"
            14: 51,    # "fence"
            15: 70,    # "vegetation"
            16: 71,    # "trunk"
            17: 72,    # "terrain"
            18: 80,    # "pole"
            19: 81,    # "traffic-sign"
        }
        return learning_map_inv
    
    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(
            segment=data_dict.pop("segment"), name=self.get_data_name(idx)
        )
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part

        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list
        return result_dict

    def __getitem__(self, idx):
        if self.test_mode:
            data_dict = self.prepare_test_data(idx)
        else: 
            data_dict = self.prepare_train_data(idx)
            if self.sources:
                data_dict_prior= self.prepare_train_data(idx+2000)
                return data_dict_prior, data_dict
        return data_dict

    def __len__(self):
        return len(self.data_list) * self.loop
