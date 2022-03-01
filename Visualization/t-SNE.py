import argparse
import os
from importlib import import_module
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from dataset import TestDataset, MaskBaseDataset
from torchvision import transforms

import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

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


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    writer = SummaryWriter(args.model_dir)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    print("Calculating inference results..")
    preds = []
    cnt = 0
    #######################t-SNE에 필요한 데이터 변수 선언 부분#######################
    total_img = torch.zeros(1,36864).to(device)
    total_label = torch.tensor([[19]]).to(device)
    trans = transforms.Compose([transforms.CenterCrop((60,80))])
    cnt=0
    """
    t-SNE 설명
    1. t-SNE 파일은 실제 inference를 하지 않고, t-SNE을 진행하여 결과물을 사진으로 출력해줍니다!
    2. "saved_fig라는 폴더를 해당 파일이 있는 디렉토리에 만들어주셔야 합니다.
    3. default는 나이대에 따라 분류한 결과를 사진으로 출력합니다. 만약 다른 기준으로 하고 싶으시면, 
        아래 current_pred 변수를 적절히 decode해주시면 됩니다! 다만, TSNE hyperparameter를 다시 튜닝하셔야 합니다.
    """
    ################################################################################

    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            #######################t-SNE에 필요한 데이터 추출 부분#######################
            #current_pred = pred.argmax(dim=-1).cpu().numpy().reshape(64,1)
            current_pred = pred.argmax(dim=-1).view(64,1) # 
            current_pred = (current_pred)%3 ### 이 부분만 바꿔서 (나이, 성별, 마스크) 분류 기준을 바꿀 수 있다!
            imgcpy = torch.clone(images).detach() #이미지 텐서들 복사
            for im in imgcpy :
                im = trans(im) # 차원 축소 연산이 부담스럽기 때문에 이미지 사이즈 축소
            imgcpy = torch.flatten(imgcpy, start_dim=1) # 이미지 별로 1차원으로 차원 축소
            total_img = torch.vstack((imgcpy, total_img)) # 이미지를 하나의 텐서로 적재
            total_label = torch.vstack((current_pred, total_label)) #예측한 레이블을 하나의 텐서로 적재
            cnt+=1
            if cnt==10: break #총 640개의 데이터만 적재 후 loop 탈출
            #############################################################################
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    #info['ans'] = preds
    #info.to_csv(os.path.join(output_dir, f'submission6.csv'), index=False)
    ####################################t-SNE 실행 부분####################################
    print("Data Loading Done! t-SNE Start...")
    total_img = total_img.detach().cpu()
    total_label = total_label.cpu().numpy()

    #(1,4800)차원의 이미지 텐서들을 TSNE를 통해 (1,2)차원으로 축소 
    # HyperParameter(learning_rate, perplexity)에 따라 사진이 크게 바뀝니다!
    img_tsne = TSNE(n_components=2, learning_rate=900, perplexity=20).fit_transform(total_img)
    #이미지 출력 및 저장 함수 호출
    plot_vecs_n_labels(img_tsne, total_label, './saved_fig/tsen.jpeg')
    print(f't-SNE Done!')
    ##########################################################################################
    

###################################이미지 출력 및 저장##################################
def plot_vecs_n_labels(v, labels, fname):
    fig = plt.figure(figsize = (10,10))
    plt.axis('off')
    sns.set_style('darkgrid')
    sns.scatterplot(x = v[:,0], y= v[:,1], hue=labels[:,0], legend='full', palette="bright")
    plt.savefig(fname)
############################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(96, 128), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='EfficientNet_MultiLabel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
