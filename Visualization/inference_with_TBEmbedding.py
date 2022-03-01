import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset

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

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
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
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )
    ##########################필요한 변수 선언#########################
    writer = SummaryWriter(args.model_dir)
    total_img = torch.zeros(1,36864).to(device)
    total_label = torch.tensor([[19]]).to(device)
    """
    TensorBoard에서 볼 때 팁
    1. PCA로 설정하시고 마우스로 둘러보시는게 제일 편합니다.
    2. t-SNE로 보시면, 학습 hyperparameter를 설정해주셔야 됩니다. -> 이게 생각보다 잘 안됩니다ㅜ
        a. Supervise 변수는 저희가 예측한 레이블을 학습에 어느정도 반영할지 인데, 0으로 해놔야, 
            중립적인 clustering 결과를 볼 수 있습니다.
        b. 개인적으로, perplexity=20, learning_rate=10 정도가 좋은 것 같습니다!
    """
    ##################################################################
    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            ##################################필요한 데이터 적재###############################   
            if len(pred) == 64 :
                current_pred = torch.clone(pred).view(64,1) 
            else :
                current_pred = torch.clone(pred).view(56,1) #마지막 batch는 크기가 다르기 때문
            current_pred = (current_pred)%3 ### 이 부분만 바꿔서 (나이, 성별, 마스크) 분류 기준을 바꿀 수 있다!
            imgcpy = torch.clone(images).detach() #이미지 텐서들 복사
            imgcpy = torch.flatten(imgcpy, start_dim=1) # 이미지 별로 1차원으로 차원 축소
            total_img = torch.vstack((imgcpy, total_img)) # 이미지를 하나의 텐서로 적재
            total_label = torch.vstack((current_pred, total_label)) #예측한 레이블을 하나의 텐서로 적재
            ##################################################################################
            preds.extend(pred.cpu().numpy())

    ##################################Adding Embedding###############################
    print("adding embedding to tensorboard")
    # Set up config. 
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # Projector는 데이터 로딩이 dynamic하지 않아서(저도 뭔진 잘 모르겠습니다..) 이렇게 dir를 지정해줘야 됩니다.
    embedding.tensor_name = "default:00000"
    embedding.metadata_path = '00000/default/metadata.tsv'
    embedding.tensor_path = '00000/default/tensors.tsv'
    projector.visualize_embeddings(args.model_dir, config)
    writer.add_embedding(total_img[:1000, :], metadata=total_label[:1000,:])
    writer.close()
    print("embedding added to tensorbaord!")
    ##################################################################################
    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'submission6.csv'), index=False)
    print(f'Inference Done!')


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
