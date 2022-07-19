import numpy as np
import os
from tqdm import tqdm
from configs import get_args

if __name__ == '__main__':
    args = get_args()
    num_class=0
    if args.dataset == 'voc12':
        img_list_path = './datasets/train_aug.txt'
        num_class=21
    elif args.dataset == 'coco':
        img_list_path = './datasets/train_coco_80_aug.txt'
        num_class=81
    f = open(img_list_path, 'r')
    img_name_list = f.read().splitlines()
    d={}
    for name in tqdm(img_name_list):
        name = name.split(' ')[0][12:-4]
        cam_dict = np.load(os.path.join(args.crf_path, name + '.npy'), allow_pickle=True).item()
        cams = np.array(list(cam_dict.values()))

        keys = np.array(list(cam_dict.keys()))
        predict = np.argmax(cams,axis=0)
        predict = keys[predict]
        h,w=predict.shape
        fg_list = []
        for i in range(1, num_class):
            fg_list.append(np.sum(predict == i)/(h * w))
        d[name] = fg_list
    np.save(os.path.join('./datasets/', args.session_name+'_fg.npy'), d)
    f.close()

