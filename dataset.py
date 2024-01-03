# 改为一通道
import torch
import torch.utils.data as data
import numpy as np
import os
import math

import cv2
from scipy.io import loadmat
import util.transform as transform



def rearrange_pts(pts):
    boxes = []
    for k in range(0, len(pts), 4):
        pts_4 = pts[k:k + 4, :]
        x_inds = np.argsort(pts_4[:, 0])
        pt_l = np.asarray(pts_4[x_inds[:2], :])
        pt_r = np.asarray(pts_4[x_inds[2:], :])
        y_inds_l = np.argsort(pt_l[:, 1])
        y_inds_r = np.argsort(pt_r[:, 1])
        tl = pt_l[y_inds_l[0], :]
        bl = pt_l[y_inds_l[1], :]
        tr = pt_r[y_inds_r[0], :]
        br = pt_r[y_inds_r[1], :]
        # boxes.append([tl, tr, bl, br])
        boxes.append(tl)
        boxes.append(tr)
        boxes.append(bl)
        boxes.append(br)
    return np.asarray(boxes, np.float32)



class pafdata(data.Dataset):

    def __init__(self, data_dir, img_ids, phase, input_h=None, input_w=None, down_ratio=4):

        self.img_ids = img_ids
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.num_classes = 17


        self.heatmap_height = self.input_h // self.down_ratio
        self.heatmap_width = self.input_w // self.down_ratio
        self.sigma = 10

        # the spread of the Gaussian peak
        self.Radius = 5

        # guassian generate
        guassian_mask = torch.zeros(2 * self.Radius, 2 * self.Radius, dtype=torch.float)
        for i in range(2 * self.Radius):
            for j in range(2 * self.Radius):
                distance = np.linalg.norm([i - self.Radius, j - self.Radius])
                if distance < self.Radius:

                    # for guassian mask
                    guassian_mask[i][j] = math.exp(-0.5 * math.pow(distance, 2) / \
                                                   math.pow(self.Radius/3, 2))

        self.guassian_mask = guassian_mask




    def generate_ground_truth(self, image, pts_2, image_h, image_w, img_name, cen_pts, ori_pts):
        channel = 17

        heatmap_17 = np.zeros((channel, image_h, image_w), dtype=np.float32)
        heatmap = np.zeros((1, image_h, image_w), dtype=np.float32)
        ind = np.zeros((17),dtype=np.int64)  # 中心点像素
        vec_ind = np.zeros((17,2), dtype=np.float32)
        reg_mask = np.zeros((17), dtype=np.uint8)


        for k in range(17):
            pts = pts_2[4 * k:4 * k + 4, :]

            cen_x, cen_y = np.mean(pts, axis=0)
            ct = np.asarray([cen_x, cen_y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            # heatmap
            margin_x_left = max(0,ct_int[0]-self.Radius)
            margin_x_right = min(image_w,ct_int[0]+self.Radius)
            margin_y_bottom = max(0,ct_int[1]-self.Radius)
            margin_y_top = min(image_h, ct_int[1] + self.Radius)
            heatmap_17[k][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.guassian_mask[0:margin_y_top - margin_y_bottom, 0:margin_x_right - margin_x_left]
            for i in range(17):
                tmp = np.stack((heatmap[0,:,:],heatmap_17[i,:,:]), axis=0)
                heatmap[0,:,:] = np.max(tmp, axis=0)

            # vector
            lc = (pts[0] + pts[2]) / 2
            rc = (pts[1] + pts[3]) / 2
            dis_c = np.sqrt((rc[0] - lc[0]) ** 2 + (rc[1] - lc[1]) ** 2)
            vec = (rc - lc) / dis_c
            vec_ind[k] = vec


            # center index
            ind[k] = ct_int[1] * image_w + ct_int[0]
            reg_mask[k] = 1





        result = {'input': torch.from_numpy(image).float(),
                  'img_name': img_name,
                  'hm': torch.from_numpy(heatmap).float(),
                  'ind': torch.from_numpy(ind),
                  'vec_ind': torch.from_numpy(vec_ind).float(),
                  'cen_pts': np.array(cen_pts,dtype=np.int32),
                  'ori_pts': ori_pts,
                  'reg_mask': torch.from_numpy(reg_mask),
                  }
        return result

    def load_annotation(self, img_name):
        pts = loadmat(os.path.join(self.data_dir, 'labels', img_name))['p2']
        pts = rearrange_pts(pts)
        return pts

    def __getitem__(self, index):
        img_name = self.img_ids[index]
        img = cv2.imread(os.path.join(self.data_dir, 'data', self.phase, img_name))
        pts = self.load_annotation(img_name)
        cen_pts = []

        if self.phase == 'train':
            # data augmentation
            datatran = transform.Compose([transform.ConvertImgFloat(),
                                          transform.PhotometricDistort(),
                                          transform.Expand(max_scale=1.5, mean=(0, 0, 0)),
                                          transform.RandomMirror_w(),
                                          transform.RandomMirror_h(),
                                          transform.Resize(h=self.input_h, w=self.input_w)])
        else:
            datatran = transform.Compose([transform.ConvertImgFloat(),
                                          transform.Resize(h=self.input_h, w=self.input_w)])

        img_new, pts_new = datatran(img.copy(), pts)
        img_new = np.clip(img_new, a_min=0., a_max=255.)
        img_new = np.transpose((img_new - 128) / 255. , (2, 0, 1))  # [c h w]
        pts_new = rearrange_pts(pts_new)

        for k in range(17):
            pts_1 = pts_new[4 * k:4 * k + 4, :]
            cen = np.mean(pts_1, axis=0)
            cen_pts.append(cen)

        pts_new = transform.rescale_pts(pts_new, down_ratio=self.down_ratio)

        dict_data = self.generate_ground_truth(image=img_new,
                                               pts_2=pts_new,
                                               image_h=self.input_h // self.down_ratio,
                                               image_w=self.input_w // self.down_ratio,
                                               img_name=img_name,
                                               cen_pts=cen_pts,
                                               ori_pts=pts)


        return dict_data

    def __len__(self):
        return len(self.img_ids)