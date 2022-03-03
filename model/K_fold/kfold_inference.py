import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

import numpy as np

from dataset import TestDataset, MaskBaseDataset

def load_model(saved_model, k_th, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, f'best_fold_{k_th}.pth')
    print(f"model path : {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    num_classes = MaskBaseDataset.num_classes  # 18

    models = [load_model(model_dir,k,num_classes, device) for k in range(1,6)]

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]


    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    oof_pred = None
    n_splits = 5

    print("Calculating inference results..")
    for k_th, model in enumerate(models):
        print(f"[{k_th + 1}] fold inferencing!")
        model.eval()
        model.cuda()
        preds = []
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)

                #Test Time Augmentation(TTA)
                pred = model(images)
                pred += model(torch.flip(images, dims =(-1,))) / 2

                preds.extend(pred.cpu().numpy())

        fold_pred = np.array(preds)

        if oof_pred is None:
            oof_pred = fold_pred / n_splits
        else :
            oof_pred += fold_pred / n_splits

    info['ans'] = np.argmax(oof_pred, axis = 1)
    info.to_csv(os.path.join(output_dir, f'submission.csv'), index=False)
    print(f'Inference Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(380, 380), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='EfficientNet_MultiLabel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/cutmix_kfold'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
