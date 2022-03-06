![image](https://user-images.githubusercontent.com/91870042/156874151-4d1c362b-bae6-4781-8064-315d12d3ba60.png)

# 마스크 착용 상태 분류 대회
카메라로 촬영한 사람 얼굴 이미지의 마스크 착용 여부를 판단하는 Task  

아시아인 사람의 얼굴을 입력으로 받으면, 해당 인물에 대해 마스크 착용상태, 성별, 나이대를 분류해주는 모델을 만들어야 한다. 마스크는 착용상태에 따라 3가지(착용, 미착용, 오착용)로 분류, 나이대는 30대 미만, 30~59, 60대 이상으로 분류하여 총 18가지의 Class로 분류할 수 있는 모델을 만들어야 하며, 평가는 **F1-Score**를 기준으로 이루어진다.

## 기간
**대회 진행**: 2022년 2월 21일 10:00 ~ 2022년 3월 3일 19:00  
**결과 발표**: 2022년 3월 3일 20:00

## 모델 학습 및 평가방법
모델의 학습에 사용되는 패키지를 `requirements.txt`를 활용해서 설치진행
```shell
pip install -r requirements.txt
--------------------------------
pip install python-dotenv
pip install efficientnet_pytorch
pip install facenet-pytorch
```
모델의 train, test를 위해서 terminal이나 jupyternotebook 환경을 사용하며, 각 과정에서 사용되는 하이퍼파라미터나 경로들을 argument로 전달 가능하다.

```shell
python train.py --optimizer Adam --model efficientnet --epochs 20
python inference.py --model_dir='./trained_model_path'
``` 
argument에 사용될 수 있는 인자는 기본적으로 다음과 같다  

- train.py
    |argument option|default|Description|
    |---|---|---|
    |log_interval|20|학습과정을 Terminal에 log로 남길 간격|
    |name|exp|학습된 모델의 정보를 저장할 파일의 이름|
    |lr_decay_step|20|lr을 감소시킬 epoch간격|
    |criterion|cross_entropy|학습에 사용되는 loss function|
    |augmentation|BaseAugmentation|학습에 사용되는 Augmentation 기법|
    |resize|[128, 96]|입력 이미지 사이즈를 조정|
    |optimizer|SGD|학습에 사용되는 Optimizer|
    |batch_size|64|[Train] dataloader에 지정할 batch_size|
    |model|BaseModel|학습에 사용될 모델|
    |val_ratio|0.2|학습데이터에서 validation 데이터로 사용할 비율|
    |valid_batch_size|1000|[Validation] dataloader에 지정할 batch_size|
    |lr|1e-3|Optimizer에 지정할 학습률|
    |seed|42|학습에 진행되는 모든 난수값을 일정하게 고정시켜주는 값|
    |epochs|1|학습에 진행되는 epoch 수|
    |dataset|MaskBaseDataset|학습에 사용되는 Dataset|
- inference.py
    |argument option|default|Description|
    |---|---|---|
    |batch_size|1000|[Inference] dataloader에 지정할 batch_size|
    |resize|(380, 380)|입력 이미지 사이즈를 조정|
    |model|BaseModel|추론과정에서 사용될 모델|
    |augmentation|CustomAugmentation|추론에 사용되는 Augmentation 기법|
    |data_dir|/opt/ml/input/data/eval|test dataset위치|
    |model_dir|./model|`.pth`파일이 저장된 모델|
    |output_dir|./output|inference결과가 저장될 위치|
<br>

## **모델 Directory 구조**
```
├── code  
│   ├── Baseline  
│   │   ├── Wandb.py  
│   │   ├── cutmix.py  
│   │   ├── dataset.py  
│   │   ├── evaluation.py  
│   │   ├── inference.py  
│   │   ├── kfold_inference.py  
│   │   ├── kfold_train.py  
│   │   ├── loss.py  
│   │   ├── model  
│   │   ├── model.py  
│   │   ├── output  
│   │   ├── requirements.txt  
│   │   └── train.py  
└── input/data
    ├── eval
    │   ├── images
    │   ├── info.csv
    │   └── submission.csv
    └── train
        ├── images
        ├── processed_train_images
        └── train.csv 

DataProcessing 
├── preprocess_image.ipynb
└── preprocess_image.py
```

- **preprocess_image**

    Train 이미지 경로(/opt/ml/input/data/train/images/)에 존재하는 이미지파일을 다음 경로로 이동
    >/opt/ml/input/data/train/processed_train_images/

    이동하면서 `{id}_{label}.jpg` 형태로 변경 ex:) **000001_1.jpg**

    label 방법은 다음과 같음
    ```
        "incorrect_mask" : "_1",
        "mask1" : "_2",
        "mask2" : "_3",
        "mask3" : "_4",
        "mask4" : "_5",
        "mask5" : "_6",
        "normal" : "_7",
    ```

<br>
<br>

# 프로젝트 진행과정
## **문제 정의**
- 풀어야 하는 문제
    - 마스크 착용 여부 (착용, 미착용, 오착용)
    - 성별 구분
    - 연령대 (0\~29, 30\~59, 60~)
- 문제의 input과 output
    - input ⇒ 사람의 이미지
    - output ⇒ 0~17
- 어떻게 사용되어질 수 있을까?
    - 마스크를 올바로 착용했는지 알아볼 수 있는 시스템

    > 카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다. 이 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적자원으로도 충분히 검사가 가능할 것입니다.
    > 

    - 마스크를 올바르게 착용하지 않는 연령대 확인가능
        - 사회적 편견 확인 가능 ( 노인은 마스크를 제대로 착용하지 않는다? )
        - 연령대별로 마스크를 잘 착용시킬 수 있도록 만들 수 있다. (인식 확인)
    - 마스크를 착용한 상태의 얼굴 인식 방법에 “성별” feature 사용
## **EDA**
1. 결측치 정보확인
2. 입력 이미지 크기 확인
3. 이미지 원본 확인
4. 학습데이터 분포 확인
    1. 성별 분포
    2. 나이대별 분포(나이 1단위, 10단위, 30단위)
    3. 분류 클래스(18개)별 분포 확인
5. 이미지에 대한 RGB & HLS(Hue, Lightness, Saturation)분포 확인
    1. HLS & RGB 값 분포 확인
    2. HLS & RGB Heatmap
    3. HLS & RGB값을 기준으로 선택한 영역 흰색 표시해보기
6. t-SNE를 통한 클러스터링 후 레이블 확인

## **Model**
- ResNet18, ResNet34로 초반 제출로 기준을 설정
- EfficientNet-b4를 사용하고 out_feature의 수를 조절하여 성능을 높힘

## **Augmentation**
- 기본 augmentation 기법으로 `torchvision.transforms`의 `Normalize`, `CentorCrop`, `GaussianBlur`, `ColorJitter`, `HorizontalFlip`을 사용
- 그 외의 성능의 향상을 위해서 다음 6가지의 augmentation 사용
    - **FaceNet**: Train을 진행할 때, 얼굴부분의 학습을 강제하기 위해서 FaceNet으로 얼굴부분만 Crop
    - **Resize**: EfficientNet-b4의 입력 크기에 맞는 이미지인 (380\*380)으로 이미지 resize진행  
    FaceNet에서도 동일하게 Padding을 두어서 380\*380으로 Face부분만 Crop이 될 수 있도록 구현
    - **Canny Edge**: 나이값의 판단에 있어서 주름이 영향을 미칠것으로 보아 주름만 부각시킬 수 있도록 선을 따주는 Canny Egde의 사용
    - **CutMix**: 적은 데이터인 train set을 좌우 이미지를 잘라붙여 더 많은 이미지를 보는 것과 같은 효과를 줄 수 있도록 CutMix구현
    - **RandomErasing**: 학습을 진행할 때, 이미지의 전체적인 부분을 볼 수 있게 도와주는 RandomErasing진행
    - **RGB to HLS & HLS Manipulation**

## **Dataset & Dataloader**
- Split by Profile : 각 이미지파일을 사람별로 구분해서 train dataset과 validation dataset으로 분할
- Weighted Sampler : 데이터 분포를 고려하여 균등하게 이미지 batch를 return
- 나이 경계값 제거 (27~29, 57~59) : 나이 클래스를 특정하기 힘든 경계에 있는 데이터를 제거하여 성능 향상 
- 특정 나이대(60대 이상) 50% 확률로 Horizentalflip : 데이터가 부족한 연령대의 데이터를 더 많이 학습시키기 위해 50% 확률로 좌우반전 시킴
- Cutmix Dataset
- 나이, 성별, 마스크 착용 여부별 Dataset

## **wandb를 사용한 결과정리 시각화**
[링크: RecSys04-ImageClassification-wandb](https://wandb.ai/recsys_04/Image_Classification/runs/2o34ygj7?workspace=user-somi)

1. validation 과정에서 틀린 데이터 확인하기 (데이터셋 이미지, label 값)
2. train 과정에서 샘플 데이터 확인하기 (학습 이미지, label 값)
3. 학습 진행과정 (loss, accuracy) 기록
4. task(mask, age, gender)별 accuracy 기록
5. MultiClass별 acccuracy 기록
6. 현재 하드웨어 상태 정보 확인 가능
7. 같은 팀원들과 학습 기록/현황 공유, 비교 가능

![image](https://user-images.githubusercontent.com/91870042/156873202-b5a40a1f-5cd9-4320-a262-2135c1d4195b.png)


## **최종 순위 및 결과**
![image](https://user-images.githubusercontent.com/91870042/156873472-27d7f406-8973-4651-8562-cd899b2e0701.png)


Public: 48팀 중 4위  
Private: 48팀 중 4위

