import numpy as np
from PIL import Image

def getPartPalette(mode):
    '''
    http://blog.csdn.net/yhl_leo/article/details/52185581
    '''
    if mode == 'color':
        pal = np.array([[0, 0, 0],
                        [128, 0, 0],
                        [0, 128, 0],
                        [128, 128, 0],
                        [0, 0, 128],
                        [128, 0, 128],
                        [0, 128, 128],
                        [128, 128, 128],
                        [64, 0, 0],
                        [192, 0, 0],
                        [64, 128, 0],
                        [192, 128, 0],
                        [64, 0, 128],
                        [192, 0, 128],
                        [64, 128, 128],
                        [192, 128, 128],
                        [0, 64, 0],
                        [128, 64, 0],
                        [0, 192, 0],
                        [128, 192, 0],
                        [0, 64, 128],
                       [128, 32, 32],
                       [32, 128, 32],
                       [128, 128, 32],
                       [32, 32, 128],
                       [128, 32, 128],
                       [32, 128, 128],
                       [128, 128, 128],
                       [64, 32, 32],
                       [192, 32, 32],
                       [64, 128, 32],
                       [192, 128, 32],
                       [64, 32, 128],
                       [192, 32, 128],
                       [64, 128, 128],
                       [192, 128, 128],
                       [32, 64, 32],
                       [128, 64, 32],
                       [32, 192, 32],
                       [128, 192, 32],
                       [32, 64, 128],
                       [128, 16, 16],
                       [16, 128, 16],
                       [128, 128, 16],
                       [16, 16, 128],
                       [128, 16, 128],
                       [16, 128, 128],
                       [128, 128, 128],
                       [64, 16, 16],
                       [192, 16, 16]], dtype='uint8').flatten()
    elif mode =='binary':
        pal = np.array([[0, 0, 0],
                        [255, 255, 255]], dtype='uint8').flatten()
    return pal

def colorize_mask(mask, mode):
    """
    :param mask: 图片大小的数值，代表不同的颜色
    :return:
    """
    new_mask = Image.fromarray(mask.astype(np.uint8), 'P')  # 将二维数组转为图像

    pal = getPartPalette(mode)
    new_mask.putpalette(pal)
    # print(new_mask.show())
    return new_mask

def labelTopng(label, img_name, mode='color'):
    '''
    convert tensor cpu label to png and save
    '''
    #label = label.numpy()             # 320 320
    label_pil = colorize_mask(label, mode)
    label_pil.save(img_name)