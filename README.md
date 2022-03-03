# level1-image-classification-level1-recsys-04
level1-image-classification-level1-recsys-04 created by GitHub Classroom

----
### **데이터 전처리 관련 파일**
**DataProcessing**


├── **preprocess_image.ipynb**


└── **preprocess_image.py**




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
---


### 설치
```
pip install python-dotenv
pip install efficientnet_pytorch
pip install facenet-pytorch
```
