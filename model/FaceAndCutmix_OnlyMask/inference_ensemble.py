import argparse
import os
from importlib import import_module

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from tensorboard.plugins import projector

def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = saved_model # os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model_AGE = load_model('./model/ensemble/best_AGE.pth', 3, device).to(device)
    model_GENDER = load_model('./model/ensemble/best_GENDER.pth', 2, device).to(device)
    model_MASK = load_model('./model/ensemble/best_MASK.pth', 3, device).to(device)

    model_AGE.eval()
    model_GENDER.eval()
    model_MASK.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, resize=args.resize)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )


    print("Calculating inference results..")
    preds = []

    with torch.no_grad():
        for idx, images in tqdm(enumerate(loader)):
            answer = []
            images = images.to(device)

            pred_AGE = model_AGE(images)
            pred_MASK = model_MASK(images)
            pred_GENDER = model_GENDER(images)

            pred_AGE = pred_AGE.argmax(dim=1)
            pred_MASK = pred_MASK.argmax(dim=1)
            pred_GENDER = pred_GENDER.argmax(dim=1)

            pred = pred_MASK * 6 + pred_GENDER * 3 + pred_AGE

            # print ('pred2 :', pred)
            preds.extend(pred.cpu().numpy())
        
    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(380, 380), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: BaseAugmentation)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
