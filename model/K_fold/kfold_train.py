import argparse
import glob
import json
import multiprocessing
import os
from pickle import FALSE
import random
import re
from importlib import import_module
from pathlib import Path
from xmlrpc.client import boolean

import matplotlib.pyplot as plt
import numpy as np
import torch, gc
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset

from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset, CustomAugmentation, CutMixDataset
from loss import create_criterion, LabelSmoothingLoss
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import KFold

from cutmix import rand_bbox

import tqdm
from tqdm.auto import tqdm

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))
    #save_dir = os.path.join(model_dir, args.name)
    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)


    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseAugmentation
    dataset = dataset_module(
        data_dir=data_dir,
        val_ratio = args.val_ratio
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)



    # -- K-fold
    n_splits = 5
    kf = KFold(n_splits=n_splits)
    
    for k_th, (train_idx, valid_idx) in enumerate(kf.split(dataset)):
        
        # -- train과 valid로 나뉘어진 index에 대해 index정보를 설정해줌
        dataset.set_index_info(train_idx,valid_idx)

        # -- data_subset
        train_set = torch.utils.data.Subset(dataset, indices = train_idx)
        val_set = torch.utils.data.Subset(dataset, indices = valid_idx)
        
        # -- data_sampler
        train_sampler = dataset.get_sampler(train_idx)
        val_sampler = dataset.get_sampler(valid_idx)
        
        # -- data_loader
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count()//2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=True,
            sampler = train_sampler,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=args.valid_batch_size,
            num_workers=multiprocessing.cpu_count()//2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=True,
            sampler = val_sampler
        )

        # -- model
        model_module = getattr(import_module("model"), args.model)  # default: BaseModel
        model = model_module(
            num_classes=num_classes
        ).to(device)

        model = torch.nn.DataParallel(model)    

        # -- loss & metric
        criterion = create_criterion(args.criterion)  # default: cross_entropy
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
        #scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=20)
        #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5)

        best_val_acc = 0
        best_val_loss = np.inf

        #file to write acc information
        file = open(os.path.join(save_dir, f"acc_info_fold_{k_th + 1}.txt"), "w") 
        # Train Start!
        print(f"============== [{k_th + 1}/{n_splits}] Fold train start! =================")
        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0

            #for progressbar
            pbar = tqdm(enumerate(train_loader), total= len(train_loader))
            for idx, train_batch in pbar:
                #figure = None
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                #start---------------------- cut Mix -----------------------------------#
                '''
                    cut Mix batch images if args.is_cutMix 'True'
                '''
                #cutmix possibility 50%
                if args.is_cutMix and random.choice([True,False]):
                    randIdx = torch.randperm(inputs.size()[0])
                    lam = 0.5 #horizental cutmix lamda
                    labels_a = labels
                    labels_b = labels[randIdx]

                    bbx1, bby1, bbx2, bby2 = rand_bbox([0,380,380], lam)
                    inputs[:,:,bbx1:bbx2, bby1:bby2] = inputs[randIdx, : , bbx1:bbx2, bby1:bby2]
                    #lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)
                    loss = criterion(outs, labels_a) * lam + criterion(outs, labels_b) * (1. - lam)
                    labels = labels_a * lam + labels_b * (1. - lam)
                #end---------------------- cut Mix -------------------------------------#
                else :
                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)
                    loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()

                # defrag cached memory
                torch.cuda.empty_cache()

                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    
                    pbar.set_postfix({
                        "epoch" : f"[{epoch + 1}/{args.epochs}]({idx + 1}/{len(train_loader)})",
                        "Train/acc" : f"{train_acc:4.2%}",
                        "Train/loss": f"{train_loss:4.4}",
                        "lr": f"{current_lr}"
                    })
                    logger.add_scalars("Train/loss",  {f'fold {k_th + 1}' :train_loss}, epoch * len(train_loader) + idx)
                    logger.add_scalars("Train/accuracy", {f'fold {k_th + 1}' : train_acc}, epoch * len(train_loader) + idx)

                    loss_value = 0
                    matches = 0

            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                ####################Adding Variables#######################
                val_correct_label = [0 for _ in range(18)]
                val_total_label = [0 for _ in range(18)]
                val_correct_label_by_age = [0,0,0]
                val_total_label_by_age = [0,0,0]
                val_correct_label_by_gender = [0,0]
                val_total_label_by_gender = [0,0]
                val_correct_label_by_mask = [0,0,0]
                val_total_label_by_mask = [0,0,0]
                ###########################################################
                figure = None
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)
                    #####################Counting Total Num and Correct Num#######################
                    for label, pred in zip(labels, preds) :
                        if (label == pred) :
                            val_correct_label[label] += 1
                        val_total_label[label] += 1
                    ##############################################################################
                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                best_val_loss = min(best_val_loss, val_loss)
                if val_acc > best_val_acc:
                    print(f"Fold [{k_th + 1}] New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/best_fold_{k_th + 1}.pth")
                    best_val_acc = val_acc
                torch.save(model.module.state_dict(), f"{save_dir}/last_fold_{k_th + 1}.pth")
                print(
                    f"Fold [{k_th + 1}] [Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                #############################Added to See Accuracy################################
                file.write(f"----------------EPOCH : {epoch}------------------\n")
                print("Accuracy By Each Label")
                for i in range(18) :
                    mask_label = (i // 6) % 3
                    gender_label = (i // 3) % 2
                    age_label = i % 3
                    accuracy = val_correct_label[i]/(val_total_label[i] + 1e-4)
                    val_total_label_by_age[age_label] += val_total_label[i]
                    val_correct_label_by_age[age_label] += val_correct_label[i]
                    val_total_label_by_gender[gender_label] += val_total_label[i]
                    val_correct_label_by_gender[gender_label] += val_correct_label[i]
                    val_total_label_by_mask[mask_label] += val_total_label[i]
                    val_correct_label_by_mask[mask_label] += val_correct_label[i]
                for i in range(3) :
                    accuracy = (val_correct_label_by_age[i]/(val_total_label_by_age[i] + 1e-4))
                    print(f"Age : {i}, Accuracy By Age : {accuracy:4.2}%")
                    file.write(f"Age : {i}, Accuracy By Age : {accuracy:4.2}%\n")
                for i in range(2) :
                    accuracy = (val_correct_label_by_gender[i]/(val_total_label_by_gender[i] + 1e-4))
                    print(f"Gender : {i}, Accuracy By Gender : {accuracy:4.2}%")
                    file.write(f"Gender : {i}, Accuracy By Gender : {accuracy:4.2}%\n")
                for i in range(3) :
                    accuracy = (val_correct_label_by_mask[i]/(val_total_label_by_mask[i] + 1e-4))
                    print(f"Mask : {i}, Accuracy By Mask : {accuracy:4.2}%")
                    file.write(f"Mask : {i}, Accuracy By Mask : {accuracy:4.2}%\n")             
                ######################################################################################
                logger.add_scalars("Val/loss", {f'fold {k_th + 1}' : val_loss}, epoch)
                logger.add_scalars("Val/accuracy", {f'fold {k_th + 1}' : val_acc}, epoch)
                logger.add_figure("results", figure, epoch)
                print()
        print(f"=================== [{k_th+1}] fold finished ==================")
        file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='KfoldCutMixDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[380, 380], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=100, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='EfficientNet_MultiLabel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.0, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=7, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='cutmix_kfold_age_recover', help='model save at {SM_MODEL_DIR}/{name}')

    #cutmix option
    parser.add_argument('--is_cutMix', type=bool, default=False, help ='options to use cutmix')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)