import numpy as np

def rand_bbox(size, lam):
    W = size[1]# C x W x H 
    H = size[2]

    #cut_rat = np.sqrt(1. - lam)
    #cut_w = np.int(W * cut_rat)
    #cut_h = np.int(H * cut_rat)

    # uniform
    #cx = np.random.randint(W)
    #cy = np.random.randint(H)

    bbx1 = np.clip(0,0,W)
    bby1 = np.clip(H // 2,0,H)
    bbx2 = np.clip(W,0,W)
    bby2 = np.clip(H,0,H)

    return bbx1, bby1, bbx2, bby2

def cutmix(im1,im2):
    #print(im1)
    bbx1, bby1, bbx2, bby2 = rand_bbox(size = im2.size(), lam = 0) #동일 라벨일 경우 ram 생략
    im1[:, bbx1:bbx2, bby1:bby2] = im2[:, bbx1:bbx2,bby1:bby2]
    return im1