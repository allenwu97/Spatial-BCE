import numpy as np
from tools import imutils
import os.path
import imageio
from tqdm import tqdm
from torch import multiprocessing
from configs import get_args
import os


def _work(process_id, args, img_name_list):
    for idx in tqdm(range(process_id, len(img_name_list), 4)):
        img_name = img_name_list[idx]
        img_name = img_name.split(' ')[0][12:-4]
        img_path = os.path.join(args.data_dir, img_name+'.jpg')
        orig_img = np.asarray(imageio.imread(img_path))
        cam_dict = np.load(os.path.join(args.cam_path, img_name + '.npy'), allow_pickle=True).item()

        cams = np.array(list(cam_dict.values()))
        keys = np.array(list(cam_dict.keys()))

        keys = np.pad(keys + 1, (1, 0), mode='constant')
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)

        cams = np.argmax(cams, axis=0)
        crf_score = imutils.crf_inference_label(orig_img, cams, n_labels=keys.shape[0], gt_prob=0.7)

        n_crf_al = dict()
        for i, key in enumerate(keys):
            n_crf_al[key] = crf_score[i]
        np.save(os.path.join(args.crf_path, img_name + '.npy'), n_crf_al)


if __name__ == '__main__':
    args = get_args()
    num_workers = 4
    img_list_path = ''
    if args.dataset == 'voc12':
        img_list_path = './datasets/train_aug.txt'
    elif args.dataset == 'coco':
        img_list_path = './datasets/train_coco_80_aug.txt'
    f = open(img_list_path, 'r')
    img_name_list = f.read().splitlines()
    if not os.path.exists(args.crf_path):
        os.mkdir(args.crf_path)
    multiprocessing.spawn(_work, nprocs=num_workers, args=(args, img_name_list), join=True)
    f.close()






