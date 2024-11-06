import math
import numpy as np
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from scipy.interpolate import InterpolatedUnivariateSpline
from hqnet.datasets.process.transforms import CLRTransforms

from copy import deepcopy

from ..registry import PROCESS


@PROCESS.register_module
class GenerateLaneLine(object):
    def __init__(self, transforms=None, cfg=None, training=True):
        self.transforms = transforms
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.max_points = cfg.max_points
        self.max_lanes = cfg.max_lanes
        self.training = training
        self.detection_pattern = cfg.detection_pattern

        if transforms is None:
            transforms = CLRTransforms(self.img_h, self.img_w)

        if transforms is not None:
            img_transforms = []
            for aug in transforms:
                p = aug['p']
                if aug['name'] != 'OneOf':
                    img_transforms.append(
                        iaa.Sometimes(p=p,
                                      then_list=getattr(
                                          iaa,
                                          aug['name'])(**aug['parameters'])))
                else:
                    img_transforms.append(
                        iaa.Sometimes(
                            p=p,
                            then_list=iaa.OneOf([
                                getattr(iaa,
                                        aug_['name'])(**aug_['parameters'])
                                for aug_ in aug['transforms']
                            ])))
        else:
            img_transforms = []
        self.transform = iaa.Sequential(img_transforms)

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))

        return lines

    def sample_lane(self, points, sample_ys):
        # this function expects the points to be sorted
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise Exception('Annotaion points have to be sorted')
        x, y = points[:, 0], points[:, 1]

        # interpolate points inside domain
        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(y[::-1],
                                              x[::-1],
                                              k=min(3,
                                                    len(points) - 1))
        domain_min_y = y.min()
        domain_max_y = y.max()
        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y)
                                            & (sample_ys <= domain_max_y)]
        assert len(sample_ys_inside_domain) > 0
        interp_xs = interp(sample_ys_inside_domain)

        # extrapolate lane to the bottom of the image with a straight line using the 2 points closest to the bottom
        two_closest_points = points[:2]
        extrap = np.polyfit(two_closest_points[:, 1],
                            two_closest_points[:, 0],
                            deg=1)
        extrap_ys = sample_ys[sample_ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)
        all_xs = np.hstack((extrap_xs, interp_xs))

        # separate between inside and outside points
        inside_mask = (all_xs >= 0) & (all_xs < self.img_w)
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]

        return xs_outside_image, xs_inside_image

    def filter_lane(self, lane):
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])

        return filtered_lane

    def transform_annotation(self, anno, img_wh=None):
        img_w, img_h = self.img_w, self.img_h

        old_lanes = anno['lanes']

        # removing lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        # sort lane points by Y (bottom to top of the image)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        # remove points with same Y (keep first occurrence)
        old_lanes = [self.filter_lane(lane) for lane in old_lanes]
        # normalize the annotation coordinates
        old_lanes = [[[
            x * self.img_w / float(img_w), y * self.img_h / float(img_h)
        ] for x, y in lane] for lane in old_lanes]
        # create tranformed annotations
        lanes = np.ones((self.max_lanes, 1 + 4 + 2 * self.max_points), dtype=np.float32) * -1e5
        lanes[:, 0] = 0
        old_lanes   = sorted(old_lanes, key=lambda x: x[0][0])
        for lane_pos, lane in enumerate(old_lanes):
            if lane_pos >= self.max_lanes:
                break
            lane = np.array(lane)
            lane_x = lane[:, 0]
            lane_y = lane[:, 1]
            lane_z = lane[:, -1]
            ind = np.argsort(lane_y, axis=0)
            lane_x = np.take_along_axis(lane_x, ind, axis=0)
            lane_y = np.take_along_axis(lane_y, ind, axis=0)
            lane_z = np.take_along_axis(lane_z, ind, axis=0)
            lane = np.stack((lane_x,lane_y), -1)
            lower_y, upper_y       = lane[0][1], lane[-1][1]
            lower_x, upper_x       = lane[0][0], lane[-1][0]
            xs                 = np.array([p[0] for p in lane]) / img_w
            ys                 = np.array([p[1] for p in lane]) / img_h
            # zs                 = (lane_z - track[2] - z_range[0]) / (z_range[1] - z_range[0])

            lanes[lane_pos, 0] = 1
            lanes[lane_pos, 1] = lower_y / img_h
            lanes[lane_pos, 2] = upper_y / img_h
            lanes[lane_pos, 3] = lower_x / img_w
            lanes[lane_pos, 4] = upper_x / img_w
            lanes[lane_pos, 5:5 + len(xs)] = xs
            lanes[lane_pos, (5 + self.max_points):(5 + self.max_points + len(ys))] = ys
            # lanes_zs[lane_pos, :len(zs)] = zs
        if self.detection_pattern == 'det2d':
            new_anno = {
                'label': lanes,
                'old_anno': anno,
            }
        elif self.detection_pattern == 'det3d':
            new_anno = {
                'label': lanes,
                'zs': lanes_zs,
                'track': track,
                'old_anno': anno,
            }

        return new_anno

    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)

        return lanes

    def __call__(self, sample):
        img_org = sample['img']
        lanes_2d = [[(point[0], point[1]) for point in lane] for lane in sample['old_anno']['raw_lanes']]
        line_strings_org = self.lane_to_linestrings(lanes_2d)
        line_strings_org = LineStringsOnImage(line_strings_org,
                                              shape=img_org.shape)


        mask = np.ones((1, img_org.shape[0], img_org.shape[1], 1), dtype=np.bool)
        for i in range(30):
            if self.training:
                img, line_strings, seg = self.transform(
                    image=img_org.copy().astype(np.uint8),
                    line_strings=line_strings_org,
                    segmentation_maps=mask)
            else:
                img, line_strings = self.transform(
                    image=img_org.copy().astype(np.uint8),
                    line_strings=line_strings_org)
            line_strings.clip_out_of_image_()
            new_anno = {'lanes': self.linestrings_to_lanes(line_strings)}

            try:
                annos = self.transform_annotation(new_anno,
                                                  img_wh=(self.img_w,
                                                          self.img_h))
                label = annos['label']

                break
            except:
                if (i + 1) == 30:
                    self.logger.critical(
                        'Transform annotation failed 30 times :(')
                    exit()

        sample['img'] = img.astype(np.float32) / 255.
        sample['lane_line'] = label
        sample['gt_points'] = new_anno['lanes']
        # sample['seg'] = seg
        sample['seg'] = np.logical_not(seg[:, :, :, 0]) if self.training else np.zeros(
            img_org.shape)

        return sample
