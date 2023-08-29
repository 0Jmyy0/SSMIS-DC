import torch
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from PIL import Image

torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


def one_hot2dist(seg: np.ndarray):
    C: int = len(seg)
    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(bool)
        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res


def BDloss(output, target):
    net_output = softmax_helper(output)
    gt_temp = target.cpu().numpy()
    with torch.no_grad():
        dist = one_hot2dist(gt_temp)
    dist = torch.from_numpy(dist).to(device)
    pc = net_output[:, 0:, ...].type(torch.float32).to(device)
    dc = dist[:, 0:, ...].type(torch.float32)
    multipled = torch.einsum("bkxyz,bkxyz->bkxyz", pc, dc)
    bd_loss = multipled.mean()
    return bd_loss


def BDloss1(output, target):
    net_output = softmax_helper(output)
    gt_temp = target.cpu().numpy()
    with torch.no_grad():
        dist = one_hot2dist(gt_temp)
    dist = torch.from_numpy(dist).to(device)
    pc = net_output[:, 0:, ...].type(torch.float32).to(device)
    dc = dist[:, 0:, ...].type(torch.float32)
    multipled = torch.einsum("bkhw,bkhw->bkhw", pc, dc)
    bd_loss = multipled.mean()
    return bd_loss


if __name__ == "__main__":
    img = Image.open('../input/basedata_new/train/imgs/hbu04_1.png')
    img = np.array(img)
    print(img.shape)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.unsqueeze(1)
    img = img.unsqueeze(2)
    print(img.shape)
    img1 = Image.open('../input/basedata_new/train/imgs/hbu04_2.png')
    img1 = np.array(img1)
    print(img1.shape)
    img1 = torch.from_numpy(img1)
    img1 = img1.unsqueeze(0)
    img1 = img1.unsqueeze(1)
    img1 = img1.unsqueeze(2)
    print(img1.shape)
    s = BDloss(img, img1)
    print(s)
