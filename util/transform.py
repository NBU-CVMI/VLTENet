import numpy as np
from numpy import random


from mycode import resize_img
from mycode import resize_pt

def rescale_pts(pts, down_ratio):
    return np.asarray(pts, np.float32)/float(down_ratio)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, pts):
        for t in self.transforms:
            img, pts = t(img, pts)
        return img, pts

class ConvertImgFloat(object):
    def __call__(self, img, pts):
        return img.astype(np.float32), pts.astype(np.float32)

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, pts):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            img *= alpha
        return img, pts





class PhotometricDistort(object):
    def __init__(self):
        self.pd = RandomContrast()
        self.rb = RandomBrightness()
        self.rln = RandomLightingNoise()

    def __call__(self, img, pts):
        img, pts = self.rb(img, pts)
        if random.randint(2):
            distort = self.pd
        else:
            distort = self.pd
        img, pts = distort(img, pts)
        img, pts = self.rln(img, pts)
        return img, pts


class Expand(object):
    def __init__(self, max_scale = 1.5, mean = (0.5, 0.5, 0.5)):
        self.mean = mean
        self.max_scale = max_scale

    def __call__(self, img, pts):
        if random.randint(2):
            return img, pts
        h,w,c = img.shape
        ratio = random.uniform(1,self.max_scale)
        y1 = random.uniform(0, h*ratio-h)
        x1 = random.uniform(0, w*ratio-w)
        if np.max(pts[:,0])+int(x1)>w-1 or np.max(pts[:,1])+int(y1)>h-1:  # keep all the pts
            return img, pts
        else:
            expand_img = np.zeros(shape=(int(h*ratio), int(w*ratio),c),dtype=img.dtype)
            expand_img[:,:,:] = self.mean
            expand_img[int(y1):int(y1+h), int(x1):int(x1+w)] = img
            pts[:, 0] += int(x1)
            pts[:, 1] += int(y1)
            return expand_img, pts



class RandomMirror_w(object):
    def __call__(self, img, pts):
        _,w,_ = img.shape
        if random.randint(2):
            img = img[:,::-1,:]
            pts[:,0] = w-pts[:,0]
        return img, pts

class RandomMirror_h(object):
    def __call__(self, img, pts):
        h,_,_ = img.shape
        if random.randint(2):
            img = img[::-1,:,:]
            pts[:,1] = h-pts[:,1]
        return img, pts




class Resize(object):
    def __init__(self, h, w):
        self.dsize = (w,h)

    def __call__(self, img, pts):
        h,w,c = img.shape
        # pts[:, 0] = pts[:, 0]/w*self.dsize[0]
        # pts[:, 1] = pts[:, 1]/h*self.dsize[1]
        # img = cv2.resize(img, dsize=self.dsize)

        img, rato = resize_img(img, self.dsize)
        pts= resize_pt(pts,rato)

        return img, np.asarray(pts)







