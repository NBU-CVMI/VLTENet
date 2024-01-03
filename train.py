
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import os
import argparse
import numpy as np
from sklearn.model_selection import KFold

from loss import LossAll
import dataset
from model import vltenet
from util.scheduler import GradualWarmupScheduler





def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=[0,1], nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--pretrained', default=None,type=str,
                        dest='pretrained', help='the path of pretrained model')
    parser.add_argument('--data_dir', type=str,
                        dest='data_dir', help='the path of data file')
    parser.add_argument('--cross_dir', type=str,
                        dest='cross_dir', help='the path of 5-fold cross-validation file')
    parser.add_argument('--input_h', default=1024, type=int,
                        dest='input_h', help='input_h')
    parser.add_argument('--input_w', default=512,  type=int,
                        dest='input_w', help='input_w')
    parser.add_argument('--down_ratio', type=int, default=4, help='down ratio')
    parser.add_argument('--epoch', type=int, default=100, help=' epoch')
    parser.add_argument('--batch_size', type=int, default=4, help=' batch_size')
    parser.add_argument('--save_path', type=str, default='', help='weights to be resumed')
    parser.add_argument('--phase', type=str, default='train', help='data directory')
    return parser.parse_args()


def construct_model():
    model = vltenet.Vltenet(pretrained=True,
                            final_kernel=1)
    model = nn.DataParallel(model).cuda()
    return model


def save_model( path, epoch, model):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch, 'state_dict': state_dict}
    torch.save(data, path)


def train_val(model, args, numi, X_train, X_test):

    train_dataset = dataset.pafdata(args.data_dir,X_train, 'train', input_h=args.input_h,
                                    input_w=args.input_w, down_ratio=args.down_ratio)

    val_dataset = dataset.pafdata(args.data_dir,X_test, 'val', input_h=args.input_h,
                                  input_w=args.input_w, down_ratio=args.down_ratio)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=6, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    criterion = LossAll()
    base_lr =1.e-5
    optimizer = torch.optim.Adam(model.parameters(), base_lr)
    after_s = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[30],gamma=0.1)
    scheduler = GradualWarmupScheduler(optimizer,10,50,after_s)

    vis_loss_train = []
    vis_loss_val = []

    for epoch in range(1,args.epoch):
        # -----------------------------train:start--------------------------
        run_train_loss = 0
        run_val_loss = 0

        run_train_losshm = 0
        run_train_lossvec = 0

        run_val_losshm = 0
        run_val_lossvec = 0

        model.train()
        lr = optimizer.param_groups[0]['lr']

        for i, data_dict in enumerate(train_loader):


            input_var = data_dict['input'].cuda()
            heatmap_var = data_dict['hm'].cuda()
            vec_ind_var = data_dict['vec_ind'].cuda()
            ind_var = data_dict['ind'].cuda()
            reg_mask_var = data_dict['reg_mask'].cuda()

            gt_batch = {'hm': heatmap_var,
                        'ind': ind_var,
                        'vec_ind': vec_ind_var,
                        'reg_mask': reg_mask_var,
                        }

            with torch.enable_grad():
                dec_dict = model(input_var)
                loss, loss_hm, loss_vec = criterion(dec_dict, gt_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            run_train_loss += loss.item()
            run_train_losshm += loss_hm.item()
            run_train_lossvec += loss_vec.item()
        scheduler.step()

        # -----------------------------train:end--------------------------



        # -----------------------------val:start--------------------------
        model.eval()
        for j, data_dict in enumerate(val_loader):

            input_var = data_dict['input'].cuda()
            heatmap_var = data_dict['hm'].cuda()
            vec_ind_var = data_dict['vec_ind'].cuda()
            ind_var = data_dict['ind'].cuda()
            reg_mask_var = data_dict['reg_mask'].cuda()


            gt_batch = {'hm': heatmap_var,
                        'ind': ind_var,
                        'vec_ind': vec_ind_var,
                        'reg_mask': reg_mask_var,
                        }

            with torch.no_grad():

                dec_dict = model(input_var)
                loss, loss_hm, loss_vec = criterion(dec_dict, gt_batch)

            run_val_loss += loss.item()
            run_val_losshm += loss_hm.item()
            run_val_lossvec += loss_vec.item()


        vis_loss_train.append(run_train_loss/len(train_loader))
        vis_loss_val.append(run_val_loss/len(val_loader))



        print('epoch:  ',epoch)
        print('lr:',lr)
        print('train loss:', vis_loss_train[-1],  'hm loss:', run_train_losshm/len(train_loader),'  vec loss:', run_train_lossvec/len(train_loader))
        print('val loss:',   vis_loss_val[-1],'   hm loss:', run_val_losshm/len(val_loader),'      vec loss:', run_val_lossvec/len(val_loader))

        # -----------------------------val:end--------------------------

        if epoch % 50 == 0 or epoch == 1:
            save_model(os.path.join(args.save_path, 'model_{}_{}.pth'.format(numi,epoch)), epoch, model)

        if len(vis_loss_val) > 1:
            if vis_loss_val[-1] < np.min(vis_loss_val[:-1]):
                save_model(os.path.join(args.save_path, 'model_las{}.pth'.format(numi)), epoch, model)






if __name__ =='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args = parse()

    img_dir = args.cross_dir
    img_ids = np.array(sorted(os.listdir(img_dir)))
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    i = 0
## 5-fold cross-validation
    for train_index, test_index in kf.split(img_ids):
        model = construct_model()
        train_val(model, args, i, img_ids[train_index], img_ids[test_index])
        i = i+1
