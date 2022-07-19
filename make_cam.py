import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn.functional as F
import os.path
from tqdm import tqdm
from torch import multiprocessing
from models import get_model
from configs import get_args
from datasets import get_dataset
from sympy import *
from scipy.optimize import fsolve
import math
from tools import imutils
from augmentations import get_aug

temp=[]

def transnpy(args):
    output_path = os.path.join(args.output_dir, args.session_name)
    model_path = os.path.join(output_path, f'{args.session_name}-{args.dataset}-final.pth')
    state_dict = torch.load(model_path)
    temp = state_dict['temperature'].tolist()
    print(temp)
    file_list = os.listdir(args.cam_path)
    for file_name in tqdm(file_list):
        if file_name[-4:] != '.npy':
            continue
        predict_file = os.path.join(args.cam_path, file_name)
        predict_dict = np.load(predict_file, allow_pickle=True).item()
        for key in predict_dict.keys():
            if type(key) != int:
               continue
            a = predict_dict['threshold'][key]
            x = symbols('x')
            results = a
            t = temp[key]
            func_np = lambdify(x, 1 / (1 + exp(-exp(t) * x)) - results, modules=['numpy'])
            r = fsolve(func_np, 0.2)
            results = r[0]
            threshold = 1 / (1 + math.exp(-results))
            predict_dict[key] = np.where(predict_dict[key] > threshold, predict_dict[key] - threshold, -1)
        predict_dict.pop('threshold')

        np.save(os.path.join(args.cam_path, file_name), predict_dict)


def _work(process_id, infer_dataset, args, model):
    with torch.no_grad():
        with torch.cuda.device(process_id % torch.cuda.device_count()):
            databin = infer_dataset[process_id]
            infer_data_loader = DataLoader(databin, shuffle=False, num_workers=0, pin_memory=False)
            tbar = tqdm(infer_data_loader)
            output_path = os.path.join(args.output_dir, args.session_name)
            model_path = os.path.join(output_path, f'{args.session_name}-{args.dataset}-final.pth')
            print(model.load_state_dict(torch.load(model_path)))
            for m in model.named_parameters():
                if 'temperature' in m[0]:
                    global temp
                    temp = m[1].tolist()
            model = model.cuda()
            model.eval()
            for iter, (img_name, img_list, label) in enumerate(tbar):
                img_name = img_name[0]
                label = label[0]
                cam_list = []
                threshold_list = []
                img_path = os.path.join(args.data_dir, img_name+'.jpg')
                orig_img = np.asarray(Image.open(img_path))
                orig_img_size = orig_img.shape[:2]
                for i in range(len(img_list)):
                    cam, threshold = model.forward_cam(img_list[i].cuda())
                    threshold = F.upsample(threshold, orig_img_size, mode='bilinear', align_corners=False)[0]
                    threshold = threshold.view(20, orig_img.shape[0] * orig_img.shape[1])
                    threshold_list.append(threshold)
                    cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy()
                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)
                    cam_list.append(cam)

                sum_cam = np.mean(cam_list, axis=0)
                sum_cam = torch.from_numpy(sum_cam)
                sum_cam = torch.sigmoid(sum_cam)

                sum_cam = sum_cam * label.view(20, 1, 1)
                max_value = torch.max(sum_cam, dim=0)[0]
                max_mask = sum_cam >= max_value
                sum_cam.masked_fill_(~max_mask, 0)
                sum_cam = sum_cam.numpy()

                norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True)+1e-8)
                norm_cam = norm_cam[:20] * label.numpy().reshape(20,1,1)

                threshold = torch.stack(threshold_list, dim=0)
                threshold = torch.mean(threshold, dim=0)
                threshold = torch.mean(threshold, dim=1)
                threshold = F.sigmoid(threshold)
                threshold = threshold.cpu().numpy()


                cam_dict = {}
                for i in range(20):
                    if label[i] > 1e-5:
                        cam_dict[i] = norm_cam[i]

                if args.adaptive_t:
                    cam_dict['threshold'] = threshold

                np.save(os.path.join(args.cam_path, img_name + '.npy'), cam_dict)



if __name__ == '__main__':
    args = get_args()
    model = get_model()
    infer_dataset = get_dataset(args.dataset, args.data_dir, transform=get_aug(args.image_size, train=False), train=False)
    num_workers = torch.cuda.device_count()
    dataset = imutils.split_dataset(infer_dataset, num_workers)
    if not os.path.exists(args.cam_path):
        os.mkdir(args.cam_path)
    multiprocessing.spawn(_work, nprocs=num_workers, args=(dataset, args, model), join=True)
    if args.adaptive_t:
        transnpy(args)

