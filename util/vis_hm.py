import numpy as np
import cv2
import torch
import torch.nn.functional as F

def norm(data):
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return (data-m)/(mx-mn)


def _nms(heat, kernel=3):
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
    keep = (hmax == heat).float()
    return heat * keep

def vis_hm(hm,path,save=False):
    hm = _nms(hm)
    cv2.imwrite(path,hm[0,0,:,:].cpu().numpy()*255)

    batch, cat, height, width = hm.size()
    topk_scores, topk_inds = torch.topk(hm.view(batch, cat, -1), 17)
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    new_y = topk_ys.squeeze()
    new_x = topk_xs.squeeze()
    new_all = torch.cat((new_x.unsqueeze(1), new_y.unsqueeze(1)), 1)
    cen_point = new_all.cpu().numpy()
    if(save):
        cen_tmp = cen_point*4
        tmp = np.zeros((height*4, width*4))
        for i in range(cen_point.shape[0]):
            cv2.circle(tmp, (int(cen_tmp[i,0]), int(cen_tmp[i,1])), 1, (255,0,0), -1, 1)
        cv2.imwrite(path, tmp)

    return cen_point




