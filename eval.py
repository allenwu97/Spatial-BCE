import os
import numpy as np
from PIL import Image
import multiprocessing
from configs import get_args

categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']


def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, threshold=1.0, if_crf=True):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))

    def compare(start, step, TP, P, T, threshold, if_crf=True):
        for idx in range(start, len(name_list), step):
            name = name_list[idx]
            name = name.split(' ')[0][12:-4]
            predict_file = os.path.join(predict_folder, '%s.npy' % name)
            predict_dict = np.load(predict_file, allow_pickle=True).item()
            h, w = list(predict_dict.values())[0].shape
            tensor = np.zeros((num_cls, h, w), np.float32)
            for key in predict_dict.keys():
                if if_crf:
                    tensor[key] = predict_dict[key]
                else:
                    tensor[key+1] = predict_dict[key]
            if not if_crf:
                tensor[0, :, :] = threshold
            predict = np.argmax(tensor, axis=0).astype(np.uint8)

            gt_file = os.path.join(gt_folder, '%s.png' % name)
            gt = np.array(Image.open(gt_file))
            cal = gt < 255
            mask = (predict == gt) * cal

            for i in range(num_cls):
                P[i].acquire()
                P[i].value += np.sum((predict == i) * cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt == i) * cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt == i) * mask)
                TP[i].release()

    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i, 8, TP, P, T, threshold, if_crf))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    for i in range(num_cls):
        IoU.append(TP[i].value / (T[i].value + P[i].value - TP[i].value + 1e-10))

    miou = np.mean(np.array(IoU))
    return miou * 100



if __name__ == '__main__':
    args = get_args()
    img_list_path = ''
    class_num = 0
    predict_dir = ''
    if_crf = False
    if args.dataset == 'voc12':
        img_list_path = './datasets/train.txt'
        class_num = 21
    elif args.dataset == 'coco':
        img_list_path = './datasets/train_coco_80_aug.txt'
        class_num = 81

    if args.type == 'cam':
        predict_dir = args.cam_path
    elif args.type == 'crf':
        predict_dir = args.crf_path
        if_crf = True

    f = open(img_list_path, 'r')
    img_name_list = f.read().splitlines()
    optimal_threshold = 0
    highest_miou = 0
    if if_crf or args.adaptive_t:
        miou = do_python_eval(predict_dir, args.gt_path, img_name_list, class_num, 0, if_crf)
        print('mIoU:%7.3f%%' % (miou))
    else:
        for i in range(20, 40, 2):
            t = i / 100.0
            miou = do_python_eval(predict_dir, args.gt_path, img_name_list, class_num, t, if_crf)
            print('threshold:%.3f    mIoU:%7.3f%%' % (t, miou))
            if miou>highest_miou:
                highest_miou = miou
                optimal_threshold = t
        print('optimal_threshold:%.3f    highest_miou:%7.3f%%' % (optimal_threshold, highest_miou))
    f.close()
