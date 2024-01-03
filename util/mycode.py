import numpy as np
import cv2

# transform: resize_pt,resize_img

def outlier_rejection(pts):

    remained_pts = []
    for i, p in enumerate(pts):


        if i == 0:
            cur_ver_center_x = p[0]
            cur_ver_width = abs(p[8] - p[2])

            next_ver_center_x = pts[i + 1][0]
            next_next_ver_center_x = pts[i + 2][0]

            if abs(next_ver_center_x - cur_ver_center_x) > (cur_ver_width / 2) and abs(
                    next_next_ver_center_x - cur_ver_center_x) > (cur_ver_width / 2):
                pass
            else:
                remained_pts.append(p)
        elif i == len(pts) - 1:
            cur_ver_center_x = p[0]
            cur_ver_width = abs(p[8] - p[2])

            pre_ver_center_x = pts[i - 1][0]
            pre_pre_ver_center_x = pts[i - 2][0]

            if abs(pre_ver_center_x - cur_ver_center_x) > (cur_ver_width / 2) and abs(
                    pre_pre_ver_center_x - cur_ver_center_x) > (cur_ver_width / 2):
                pass
            else:
                remained_pts.append(p)
        else:
            cur_ver_center_x = p[0]
            cur_ver_width = abs(p[8] - p[2])

            pre_ver_center_x = pts[i - 1][0]
            next_ver_center_x = pts[i + 1][0]

            if abs(pre_ver_center_x - cur_ver_center_x) > (cur_ver_width / 2) and abs(
                    next_ver_center_x - cur_ver_center_x) > (cur_ver_width / 2):
                pass
            else:
                remained_pts.append(p)

    if len(remained_pts) < 17:
        missing_number = 17 - len(remained_pts)
        print(f'[WARNING] number of vertebra less than 17 ! missing numbers is {missing_number}')
        remained_pts = remained_pts + remained_pts[-missing_number:]

    remained_pts = np.array(remained_pts)

    return remained_pts



def resize_img(src, dst_wh):

    src = src[:, :, 0]

    # If not HWC image, quit
    if len(src.shape) != 2:
        raise ValueError("scr is not gray scale")



    sh = src.shape[0]
    sw = src.shape[1]
    dh = dst_wh[1]
    dw = dst_wh[0]
    # ratio: W/H
    ratio_src = sw / sh
    ratio_dst = dw / dh

    if ratio_src >= ratio_dst:
        # resize by W
        resize_ratio = dw / sw
        nw = dw
        nh = int(sh * resize_ratio)
    else:
        # resize by H
        resize_ratio = dh / sh
        nw = int(sw * resize_ratio)
        nh = dh

    resized_img = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_CUBIC)
    black = np.zeros([dh, dw], dtype=np.uint8)
    left = (dw-nw)//2
    top = (dh-nh)//2
    black[top: top+nh, left: left+nw] = resized_img[...]
    result = black

    resize_record = (left, top, resize_ratio)

    tmp = np.zeros((dh, dw, 3))
    tmp[:, :, 0] = result
    tmp[:, :, 1] = result
    tmp[:, :, 2] = result
    result = tmp

    return result, resize_record


def resize_pt(xy, resize_record):

    left, top, ratio = resize_record
    xy[:, 0] = xy[:, 0] * ratio + left
    xy[:, 1] = xy[:, 1] * ratio + top
    return xy


def fix_17(img, pts):
    y = pts[:, 1]
    ymin = int(np.min(y)) - 20
    ymax = int(np.max(y)) + 30
    if (ymin < 0):
        ymin = 0
    if (ymax > img.shape[0]):
        ymax = img.shape[0]
    img[0:ymin, :, :] = 0
    crop = img[:ymax, :, :]
    img = crop
    return img