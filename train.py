import os
import torch
import torch.nn as nn
from tqdm import tqdm
from configs import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, PlotLogger
from datasets import get_dataset
from optimizers import PolynomialLR
from tensorboardX import SummaryWriter
from loss_functions import KLLoss, Spatial_BCE_Loss



not_training = ['conv1a', 'b2', 'b2_1', 'b2_2']

def get_params(model, key):
    # For Dilated FCN
    if key == "1x":
        for m in model.named_modules():
            if ("fc" not in m[0]) and ('threshold' not in m[0]) and ("mask_layer" not in m[0]):
                if isinstance(m[1], nn.Conv2d):
                    if m[0].split('.')[-1] not in not_training and m[0].split('.')[0] not in not_training:
                        for p in m[1].parameters():
                            print(1, m[0])
                            yield p

    if key == "10x":
        for m in model.named_modules():
            if ("fc" in m[0]) or ('threshold' in m[0]) or ("mask_layer" in m[0]):
                if isinstance(m[1], nn.Conv2d):
                    print(10, m[0])
                    yield m[1].weight

    if key == "temperature":
        for name, param in model.named_parameters():
            if 'temperature' in name:
                print(10, name)
                yield param


def main(args):
    train_set = get_dataset(
        args.dataset,
        args.data_dir,
        transform=get_aug(args.image_size, train=True),
        train=True,
        fg_path=args.fg_path
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    writer = SummaryWriter('runs/'+args.session_name,flush_secs=10)

    model = get_model()
    model = model.to(args.device)
    model_dict = model.state_dict()
    if args.pretrain:
        if args.pretrain[-7:] == '.params':
            import models.resnet38
            save_model = models.resnet38.convert_mxnet_to_torch(args.cls_pretrain)
        else:
            save_model = torch.load(args.pretrain)
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # state_dict = save_model
        print(model.load_state_dict(state_dict, strict=False))

    if args.resume:
        print(args.resume)
        save_model = torch.load(args.resume)
        save_model = save_model["state_dict"]
        print(model.load_state_dict(save_model, strict=False))

    model = torch.nn.DataParallel(model)

    optimizer = torch.optim.SGD(
        params=[
            {
                "params": get_params(model.module, key="1x"),
                "lr": args.base_lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": get_params(model.module, key="10x"),
                "lr": 10 * args.base_lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": get_params(model.module, key="temperature"),
                "lr": 10 * args.base_lr,
                "weight_decay": 0.0,
            },
        ],
        momentum=args.momentum,
    )

    lr_scheduler = PolynomialLR(
        optimizer=optimizer,
        step_size=1,
        iter_max=args.num_epochs * len(train_loader),
        power=0.9,
    )

    start_epoch = 0

    loss_meter = AverageMeter(name='Loss')
    spatial_bce_loss_meter = AverageMeter(name='Spatial_BCE_Loss')
    kl_loss_meter = AverageMeter(name='KLLoss')
    plot_logger = PlotLogger(params=['epoch', 'lr', 'loss'])
    # Start training
    global_progress = tqdm(range(start_epoch, args.num_epochs), desc='Training')
    global_step = start_epoch * len(train_loader)

    spatial_bce_loss_function = Spatial_BCE_Loss().cuda()
    klloss_function = KLLoss().cuda()

    output_path = os.path.join(args.output_dir, args.session_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for epoch in global_progress:
        loss_meter.reset()
        spatial_bce_loss_meter.reset()
        kl_loss_meter.reset()
        model.train()
        print('epoch', epoch)
        local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.num_epochs}', disable=args.hide_progress)
        for idx, (image_names, images, labels, fg) in enumerate(local_progress):
            cls_logit, threshold = model.forward(images.to(args.device))
            labels = labels.cuda()
            spatial_bce_loss = spatial_bce_loss_function(cls_logit, labels, threshold_p=threshold, fg=fg, iter=global_step)
            klloss = klloss_function(cls_logit, labels, fg.cuda())
            loss = spatial_bce_loss + klloss

            global_step +=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            spatial_bce_loss_meter.update(spatial_bce_loss.item())
            kl_loss_meter.update(klloss.item())
            lr_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            local_progress.set_postfix({'lr': lr, "loss": loss_meter.val, "spatial_bce_loss": spatial_bce_loss_meter.val, "klloss": kl_loss_meter.val})
            plot_logger.update({'epoch': epoch, 'lr': lr, 'loss': loss_meter.val})
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Spatial-BCE/Loss', spatial_bce_loss.item(), global_step)
            writer.add_scalar('KL/Loss', klloss.item(), global_step)
            writer.add_scalar('Lr', lr, global_step)
        global_progress.set_postfix({"epoch": epoch, "loss_avg": loss_meter.avg, "spatial_bce_loss": spatial_bce_loss_meter.avg, "klloss": kl_loss_meter.avg})
        # # plot_logger.save(os.path.join(args.output_dir, 'logger.svg'))
        print('Iter:%5d/%5d' % (epoch+1, args.num_epochs),
              'Loss:%.4f' % (loss_meter.avg))

        # Save checkpoint
        model_path = os.path.join(output_path, f'{args.session_name}-{args.dataset}-epoch{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),  # will double the checkpoint file size
            'lr_scheduler': lr_scheduler.state_dict(),
            'args': args,
            'loss_meter': loss_meter,
            'plot_logger': plot_logger
        }, model_path)
        print(f"Model saved to {model_path}")
        for m in model.named_parameters():
            if 'temperature' in m[0]:
                print(m[0], m[1])

    torch.save(model.module.state_dict(), os.path.join(output_path, f'{args.session_name}-{args.dataset}-final.pth'))
    writer.close()

if __name__ == "__main__":
    main(args=get_args())
