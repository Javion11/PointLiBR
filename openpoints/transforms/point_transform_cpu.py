import numpy as np
import torch
import scipy
from collections.abc import Sequence
from .point_transformer_gpu import DataTransforms
from ..dataset.data_util import fnv_hash_vec, ravel_hash_vec


@DataTransforms.register_module()
class PointsToTensor(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):  
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            if type(data[key]) is np.ndarray:
                data[key] = torch.from_numpy(data[key])
            if key in ['y', 'label', 'labels', 'index','indices', 'cls']: 
                data[key] = data[key].to(torch.long)
            elif key is not 'name':
                data[key] = data[key].to(torch.float32)
        return data


@DataTransforms.register_module()
class RandomRotate(object):
    def __init__(self,
                 angle=None,
                 center=None,
                 axis='z',
                 always_apply=False,
                 p=0.5):
        self.angle = [-1, 1] if (angle is None) or (angle==[0,0,1])  else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if np.random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == 'x':
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == 'y':
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == 'z':
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "pos" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["pos"].min(axis=0)
                x_max, y_max, z_max = data_dict["pos"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["pos"] -= center
            data_dict["pos"] = np.dot(data_dict["pos"], np.transpose(rot_t))
            data_dict["pos"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict

@DataTransforms.register_module()
class RandomScale(object):
    def __init__(self, scale=[0.9, 1.1], anisotropic=False, **kwargs):
        self.scale = scale
        self.anisotropic = anisotropic

    def __call__(self, data):
        scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
        data['pos'] *= scale
        return data


@DataTransforms.register_module()
class RandomShift(object):
    def __init__(self, shift=[0.2, 0.2, 0], **kwargs):
        self.shift = shift

    def __call__(self, data):
        shift_x = np.random.uniform(-self.shift[0], self.shift[0])
        shift_y = np.random.uniform(-self.shift[1], self.shift[1])
        shift_z = np.random.uniform(-self.shift[2], self.shift[2])
        data['pos'] += [shift_x, shift_y, shift_z]
        return data


@DataTransforms.register_module()
class RandomScaleAndTranslate(object):
    def __init__(self,
                 scale=[0.9, 1.1],
                 shift=[0.2, 0.2, 0],
                 scale_xyz=[1, 1, 1],
                 **kwargs):
        self.scale = scale
        self.scale_xyz = scale_xyz
        self.shift = shift

    def __call__(self, data):
        scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
        scale *= self.scale_xyz
        shift_x = np.random.uniform(-self.shift[0], self.shift[0])
        shift_y = np.random.uniform(-self.shift[1], self.shift[1])
        shift_z = np.random.uniform(-self.shift[2], self.shift[2])
        data['pos'] = np.add(np.multiply(data['pos'], scale), [shift_x, shift_y, shift_z])
        return data


@DataTransforms.register_module()
class RandomJitter(object):
    def __init__(self, jitter_sigma=0.01, jitter_clip=0.05, **kwargs):
        self.sigma = jitter_sigma
        self.clip = jitter_clip

    def __call__(self, data):
        jitter = np.clip(self.sigma * np.random.randn(data['pos'].shape[0], 3), -1 * self.clip, self.clip)
        data['pos'] += jitter
        return data


@DataTransforms.register_module()
class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None, **kwargs):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data):
        if np.random.rand() < self.p:
            lo = np.min(data['x'][:, :3], 0, keepdims=True)
            hi = np.max(data['x'][:, :3], 0, keepdims=True)
            if hi.max()>1.0:
                scale = 255.0 / (hi - lo)
            else:
                scale = 1.0 / (hi - lo)
            contrast_feat = (data['x'][:, :3] - lo) * scale
            blend_factor = np.random.rand() if self.blend_factor is None else self.blend_factor
            data['x'][:, :3] = (1 - blend_factor) * data['x'][:, :3] + blend_factor * contrast_feat
        return data


@DataTransforms.register_module()
class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05, **kwargs):
        self.p = p
        self.ratio = ratio

    def __call__(self, data):
        if np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            data['x'][:, :3] = np.clip(tr + data['x'][:, :3], 0, 255)
        return data


@DataTransforms.register_module()
class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005, **kwargs):
        self.p = p
        self.std = std

    def __call__(self, data):
        if np.random.rand() < self.p:
            noise = np.random.randn(data['x'].shape[0], 3)
            noise *= self.std * 255
            data['x'][:, :3] = np.clip(noise + data['x'][:, :3], 0, 255)
        return data


@DataTransforms.register_module()
class HueSaturationTranslation(object):
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        
        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __init__(self, hue_max=0.5, saturation_max=0.2, **kwargs):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, data):
        # Assume feat[:, :3] is rgb
        hsv = HueSaturationTranslation.rgb_to_hsv(data['x'][:, :3])
        hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
        sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        data['x'][:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)
        return data


@DataTransforms.register_module()
class RandomDropColor(object):
    def __init__(self, color_drop=0.2, **kwargs):
        self.p = color_drop

    def __call__(self, data):
        if np.random.rand() < self.p:
            data['x'][:, :3] = 0
        return data


#########################################PTV2####################################################
@DataTransforms.register_module()
class CenterShift(object):
    def __init__(self, apply_z=True, **kwargs):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "pos" in data_dict.keys():
            x_min, y_min, z_min = data_dict["pos"].min(axis=0)
            x_max, y_max, _ = data_dict["pos"].max(axis=0)
            if self.apply_z:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
            else:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
            data_dict["pos"] -= shift
        return data_dict


@DataTransforms.register_module()
class RandomDropoutCPU(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5, **kwargs):
        """
            upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if np.random.random() < self.dropout_application_ratio:
            n = len(data_dict["pos"])
            idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
            if "pos" in data_dict.keys():
                data_dict["pos"] = data_dict["pos"][idx]
            if "x" in data_dict.keys():
                data_dict["x"] = data_dict["x"][idx]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx]
            if "y" in data_dict.keys():
                data_dict["y"] = data_dict["y"][idx] \
                    if len(data_dict["y"]) != 1 else data_dict["y"]
        return data_dict


@DataTransforms.register_module()
class ElasticDistortion(object):
    def __init__(self, distortion_params=None, **kwargs):
        self.distortion_params = [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
        blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
        blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity *
                                       (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=False, fill_value=0)
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, data_dict):
        if "pos" in data_dict.keys() and self.distortion_params is not None:
            if np.random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    data_dict["pos"] = self.elastic_distortion(data_dict["pos"], granularity, magnitude)
        return data_dict


@DataTransforms.register_module()
class Voxelize(object):
    def __init__(self,
                 voxel_size=0.05,
                 hash_type="fnv",
                 mode='train',
                 keys=("pos", "normal", "x", "y"),
                 **kwargs):
        self.voxel_size = voxel_size
        self.hash = fnv_hash_vec if hash_type == "fnv" else ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys

    def __call__(self, data_dict):
        assert "pos" in data_dict.keys()
        discrete_coord = np.floor(data_dict["pos"] / np.array(self.voxel_size)).astype(np.int)
        key = self.hash(discrete_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, count = np.unique(key_sort, return_counts=True)
        if self.mode == 'train':  # train mode
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_unique = idx_sort[idx_select]
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        elif self.mode == 'test':  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = dict(index=idx_part)
                # TODO to be more robust
                for key in self.keys:
                    data_part[key] = data_dict[key][idx_part]
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError


@DataTransforms.register_module()
class SphereCrop(object):
    def __init__(self, point_max=80000, sample_rate=None, mode="random", **kwargs):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center", "all"]
        self.mode = mode

    def __call__(self, data_dict):
        point_max = int(self.sample_rate * data_dict["pos"].shape[0]) \
            if self.sample_rate is not None else self.point_max

        assert "pos" in data_dict.keys()
        if self.mode == "all":
            # TODO: Optimize
            if "index" not in data_dict.keys():
                data_dict["index"] = np.arange(data_dict["pos"].shape[0])
            data_part_list = []
            if data_dict["pos"].shape[0] > point_max:
                coord_p, idx_uni = np.random.rand(data_dict["pos"].shape[0]) * 1e-3, np.array([])
                while idx_uni.size != data_dict["index"].shape[0]:
                    init_idx = np.argmin(coord_p)
                    dist2 = np.sum(np.power(data_dict["pos"] - data_dict["pos"][init_idx], 2), 1)
                    idx_crop = np.argsort(dist2)[:point_max]

                    data_crop_dict = dict()
                    if "pos" in data_dict.keys():
                        data_crop_dict["pos"] = data_dict["pos"][idx_crop]
                    if "discrete_coord" in data_dict.keys():
                        data_crop_dict["discrete_coord"] = data_dict["discrete_coord"][idx_crop]
                    if "normal" in data_dict.keys():
                        data_crop_dict["normal"] = data_dict["normal"][idx_crop]
                    if "x" in data_dict.keys():
                        data_crop_dict["x"] = data_dict["x"][idx_crop]
                    data_crop_dict["weight"] = dist2[idx_crop]
                    data_crop_dict["index"] = data_dict["index"][idx_crop]
                    data_part_list.append(data_crop_dict)

                    delta = np.square(1 - data_crop_dict["weight"] / np.max(data_crop_dict["weight"]))
                    coord_p[idx_crop] += delta
                    idx_uni = np.unique(np.concatenate((idx_uni, data_crop_dict["index"])))
            else:
                data_crop_dict = data_dict.copy()
                data_crop_dict["weight"] = np.zeros(data_dict["pos"].shape[0])
                data_crop_dict["index"] = data_dict["index"]
                data_part_list.append(data_crop_dict)
            return data_part_list
        # mode is "random" or "center"
        elif data_dict["pos"].shape[0] > point_max:
            if self.mode == "random":
                center = data_dict["pos"][np.random.randint(data_dict["pos"].shape[0])]
            elif self.mode == "center":
                center = data_dict["pos"][data_dict["pos"].shape[0] // 2]
            else:
                raise NotImplementedError
            idx_crop = np.argsort(np.sum(np.square(data_dict["pos"] - center), 1))[:point_max]
            for key in data_dict.keys():
                data_dict[key] = data_dict[key][idx_crop]
        return data_dict


@DataTransforms.register_module()
class NormalizeColor(object):
    def __init__(self,
                 mean_std=True,
                 color_mean=[0.47793125906962, 0.4303257521323044, 0.3749598901421883],
                 color_std=[0.2834475483823543, 0.27566157565723015, 0.27018971370874995],
                 **kwargs):
        self.mean_std = mean_std
        self.color_mean = np.array(color_mean, dtype=np.float)
        self.color_std = np.array(color_std, dtype=np.float)
    def __call__(self, data_dict):
        if data_dict['x'][:, :3].max() > 1:
            data_dict['x'][:, :3] /= 255.
        if self.mean_std:
            data_dict['x'][:, :3] = (data_dict['x'][:, :3] - self.color_mean) / self.color_std
        return data_dict


@DataTransforms.register_module()
class ShufflePoint(object):
    def __call__(self, data_dict):
        assert "pos" in data_dict.keys()
        shuffle_index = np.arange(data_dict["pos"].shape[0])
        np.random.shuffle(shuffle_index)
        if "pos" in data_dict.keys():
            data_dict["pos"] = data_dict["pos"][shuffle_index]
        if "discrete_coord" in data_dict.keys():
            data_dict["discrete_coord"] = data_dict["discrete_coord"][shuffle_index]
        if "x" in data_dict.keys():
            data_dict["x"] = data_dict["x"][shuffle_index]
        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"][shuffle_index]
        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"][shuffle_index]
        if "y" in data_dict.keys():
            data_dict["y"] = data_dict["y"][shuffle_index] \
                if len(data_dict["y"]) != 1 else data_dict["y"]
        return data_dict


#########################################SemanticKITTI SPVCNN####################################################
@DataTransforms.register_module()
class PointClip(object):
    def __init__(self, point_cloud_range=(-80, -80, -3, 80, 80, 1)):
        self.point_cloud_range = point_cloud_range

    def __call__(self, data_dict):
        if "pos" in data_dict.keys():
            data_dict["pos"] = np.clip(
                data_dict["pos"],
                a_min=self.point_cloud_range[:3],
                a_max=self.point_cloud_range[3:],
            )
        return data_dict


@DataTransforms.register_module()
class GridSample(object):
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        keys=("pos", "x", "normal", "y"),
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
    ):
        self.grid_size = grid_size
        self.hash = fnv_hash_vec if hash_type == "fnv" else ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement

    def __call__(self, data_dict):
        assert "pos" in data_dict.keys()
        scaled_coord = data_dict["pos"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = np.unique(
                    np.append(idx_unique, data_dict["sampled_index"])
                )
                mask = np.zeros_like(data_dict["y"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        elif self.mode == "test":  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = dict(index=idx_part)
                if self.return_inverse:
                    data_dict["inverse"] = np.zeros_like(inverse)
                    data_dict["inverse"][idx_sort] = inverse
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part]
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                if self.return_displacement:
                    displacement = (
                        scaled_coord - grid_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = np.sum(
                            displacement * data_dict["normal"], axis=-1, keepdims=True
                        )
                    data_dict["displacement"] = displacement[idx_part]
                for key in data_dict.keys():
                    if key in self.keys:
                        data_part[key] = data_dict[key][idx_part]
                    else:
                        data_part[key] = data_dict[key]
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError


@DataTransforms.register_module()
class Collect(object):
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        """
        e.g. Collect(keys=[pos], feat_keys=[pos, strength])
        """
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="pos")
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data
