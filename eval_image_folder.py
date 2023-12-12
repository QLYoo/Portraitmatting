import torch
import numpy as np
import model
import os
import cv2
from pymatting import *
return_fg=False
maxhw = 640
folderpath = './Images/'
outpath = './Outs/'
os.makedirs(outpath, exist_ok=True)
if __name__ == '__main__':
    matmodel = model.Basicmatting()
    matmodel.load_state_dict(torch.load('res34.ckpt', map_location='cpu'))
    matmodel.eval()
    matmodel = matmodel.cuda()
    files = os.listdir(folderpath)
    for file in files:
        im = cv2.imread(folderpath + file)
        im_copy=im
        fh, fw, _ = im.shape
        fh, fw, _ = im.shape
        if fh > fw:
            _fw = maxhw
            _fh = int((maxhw / fw * fh - 1) // 32 + 1) * 32
        else:
            _fh = maxhw
            _fw = int((maxhw / fh * fw - 1) // 32 + 1) * 32
        im = cv2.resize(im, (_fw, _fh), interpolation=cv2.INTER_CUBIC)
        im_ = im
        im = im[:, :, ::-1]
        im = im / 255.
        im = np.transpose(im, [2, 0, 1])
        im = im[np.newaxis, :, :, :].astype(np.float32)
        im = torch.from_numpy(im).cuda()
        Tm = im
        alls = torch.cat([Tm], 1)
        with torch.no_grad():
            alpha, tri = matmodel(alls)
        alpha = (torch.clamp((alpha + 1) * 127.5, 0, 255))
        tri = torch.argmax(tri, 1, keepdim=True)
        alpha = alpha * (tri == 1) + (tri == 2) * 255
        alpha = alpha[0, 0].cpu().numpy()
        alpha = alpha.astype(np.uint8)
        alpha = cv2.resize(alpha, (fw, fh), interpolation=cv2.INTER_CUBIC)
        if return_fg:
            fg=estimate_foreground(im_copy/255., alpha/255.)*255.
            fg=fg.astype(np.uint8)
            image=np.concatenate((fg,alpha[:, :, np.newaxis]),axis=2)
            cv2.imwrite(outpath + file[:-3] + 'png', image)
        else:
            cv2.imwrite(outpath + file[:-3] + 'png', alpha)
