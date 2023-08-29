import logging
import os
import random
import pandas as pd
import shutil
import sys
from glob import glob
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss, BCELoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch import nn
from parameter import get_parameters
from utils.brats2019 import (BraTS2019, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from utils import ramps
from losses import losses
from val_3D import test_all_case
from networks.model_3D import  models_3D
from utils.boundary_loss import BDloss
from utils.losses import losses



torch.cuda.set_device(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = get_parameters()

def CutoutAbs(volume, ratio=0.5):
    length_w = int(ratio * volume.shape[0])
    length_d = int(ratio * volume.shape[1])
    length_h = int(ratio * volume.shape[2])
    start_w = random.randint(0, volume.shape[0])
    start_d = random.randint(0, volume.shape[1])
    start_h = random.randint(0, volume.shape[2])
    end_w = (start_w + length_w) if (start_w + length_w) < volume.shape[0] else (volume.shape[0] - 1)
    end_d = (start_d + length_d) if (start_d + length_d) < volume.shape[1] else (volume.shape[1] - 1)
    end_h = (start_h + length_h) if (start_h + length_h) < volume.shape[2] else (volume.shape[2] - 1)
    new_volume = volume.copy()
    del volume
    new_volume[start_w:end_w, start_d:end_d, start_h:end_h, :] = 0
    return new_volume

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr 
    num_classes = args.num_classes
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False): 
        model = models_3D(in_channels=1, n_classes=2).to(device)
        if ema:
            for param in model.parameters():
                param.detach_()  
        return model

    model = create_model() 
    ema_model = create_model(ema=True)  

    def worker_init_fn(worker_id): 
        random.seed(args.seed + worker_id)

    db_train = BraTS2019(base_dir=train_data_path,
                         split='train',
                         num=None,
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ]))  

    total_slices = len(db_train)  
    labeled_slice = args.labeled_num  

    labeled_idxs = list(range(0, labeled_slice))  
    unlabeled_idxs = list(range(labeled_slice, total_slices))  

    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss1(2)

    logs = pd.DataFrame(index=[], columns=['iter_num', 'performance', 'mean_hd95'])
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch = volume_batch.to(device, dtype=torch.float32)
            mask_type = torch.float32 if num_classes == 1 else torch.long
            label_batch = label_batch.to(device, dtype=mask_type)

            unlabeled_volume_batch = volume_batch[args.labeled_bs:]  

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise  

            unlabeled_volume_batch1 = unlabeled_volume_batch
            unlabeled_volume_batch1 = unlabeled_volume_batch1.cpu().numpy()
            unlabeled_volume_batch1 = unlabeled_volume_batch1.transpose((2, 3, 4, 0, 1))
            unlabeled_volume_batch1 = CutoutAbs(unlabeled_volume_batch1, 0.5)
            unlabeled_volume_batch1 = unlabeled_volume_batch1.transpose((3, 4, 0, 1, 2))
            unlabeled_volume_batch1 = torch.from_numpy(unlabeled_volume_batch1).to(device)

            unlabeled_volume_batch1 = unlabeled_volume_batch1

            outputs1, bd = model(volume_batch) 

            outputs2, outputs3, _ = model(unlabeled_volume_batch1)  

            outputs_soft1 = torch.softmax(outputs1, dim=1)
            bd = torch.softmax(bd, dim=1)

            outputs_soft2 = torch.softmax(outputs2, dim=1)  
            outputs_soft3 = torch.softmax(outputs3, dim=1)
            with torch.no_grad():

                ema_output, ema_output1, _ = ema_model(ema_inputs)
                ema_output2, ema_output3, _ = ema_model(unlabeled_volume_batch1)

                ema_output_soft = torch.softmax(ema_output, dim=1)
                ema_output_soft1 = torch.softmax(ema_output1, dim=1)

                ema_output_soft2 = torch.softmax(ema_output2, dim=1)
                ema_output_soft3 = torch.softmax(ema_output3, dim=1)

            loss_ce = ce_loss(outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs])
            loss_dice = dice_loss(outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            label_batch1 = label_batch[:args.labeled_bs].unsqueeze(1)
            loss_bd = BDloss(bd[:args.labeled_bs], label_batch1[:args.labeled_bs])
            supervised_loss = 0.5 * (loss_dice + loss_ce) + 0.002 * loss_bd

            consistency_weight = get_current_consistency_weight(iter_num//150)  

            consistency_loss = torch.mean((outputs_soft[args.labeled_bs:] - ema_output_soft) ** 2)   
            consistency_loss1 = torch.mean((outputs_soft1[args.labeled_bs:] - ema_output_soft1) ** 2)

            consistency_loss2 = torch.mean((outputs_soft2 - ema_output_soft2) ** 2)  
            consistency_loss3 = torch.mean((outputs_soft3 - ema_output_soft3) ** 2)

            consistency_loss_sum_ruo = consistency_loss + consistency_loss1  
            consistency_loss_sum_qiang = consistency_loss2 + consistency_loss3  
            consistency_loss_sum = consistency_loss_sum_ruo + consistency_loss_sum_qiang

            loss = supervised_loss + consistency_weight * consistency_loss_sum

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))


            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                avg_metric = test_all_case(
                    model, args.root_path, test_list="val.txt", num_classes=2, patch_size=args.patch_size,
                    stride_xy=64, stride_z=64)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score',
                                  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[0, 1], iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
                model.train()
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)  
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../models/{}/{}_{}_labeled/{}".format(   
        'brats2019', args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):  
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, 
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  
    logging.info(str(args))  
    train(args, snapshot_path) 
