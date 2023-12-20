import torch
import numpy as np
import model
import cv2
from pymatting import *
maxhw = 768
high_quality = False
background='bg.jpg'
def resize_image(im):
    fh, fw, _ = im.shape
    if fh > fw:
        _fw = maxhw
        _fh = int((maxhw / fw * fh - 1) // 32 + 1) * 32
    else:
        _fh = maxhw
        _fw = int((maxhw / fh * fw - 1) // 32 + 1) * 32
    im = cv2.resize(im, (_fw, _fh), interpolation=cv2.INTER_LANCZOS4)
    return im

if __name__ == '__main__':
    matmodel = model.BasicmattingNEO()
    matmodel.load_state_dict(torch.load('swinmat.ckpt', map_location='cpu'))
    matmodel.eval()
    matmodel = matmodel.cuda()
    cap = cv2.VideoCapture(0)
    bg=cv2.imread(background)
    flag = True
    while True:
        ret, frame = cap.read()
        if flag:
            bg=cv2.resize(bg,(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_LANCZOS4)
            flag=False

        if not ret:
            print("Failed to capture frame")
            break
        im = resize_image(frame)
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
        alpha = (torch.clamp(alpha, 0, 1))*255.
        tri = torch.argmax(tri, 1, keepdim=True)
        alpha = alpha * (tri == 1) + (tri == 2) * 255
        alpha = alpha[0, 0].cpu().numpy()
        alpha = alpha.astype(np.uint8)
        alpha = cv2.resize(alpha, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        if high_quality:
            fg=estimate_foreground(frame/255., alpha/255.)*255.
        else:
            fg=frame*1.
        alpha=alpha[:, :, None]
        newimage=fg*(alpha/255.)+bg*(1.-alpha/255.)
        newimage=np.array(newimage,dtype=np.uint8)
        cv2.imshow('New Image', newimage)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()