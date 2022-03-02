import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import *

from facenet_pytorch import MTCNN
import PIL

from cutmix import cutmix						 

#이미지 확장자의 종류를 담고 있는 리스트
IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]

#endswith() - 지정한 점미사로 끝나면 True, 아니면 False를 반환
#amy()
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



class BaseAugmentation:
    def __init__(self):
        self.transform = transforms.Compose([
            Resize([380,380], Image.BILINEAR),
            ToTensor(),
            #FaceDetect((380,380)),
            Normalize(mean=(0.527, 0.465, 0.435), std=(0.24, 0.242, 0.243)),
										   
        ])

    def __call__(self, image):
        return self.transform(image)

class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([

            Resize(resize, Image.BILINEAR),
            ToTensor(),
            RandomErasing(p=1, scale=(0.025,0.025), ratio=(0.5,1)),
            Normalize(mean=mean, std=std)
        ])

    def __call__(self, image):
        return self.transform(image)

class FactNetAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            ToTensor(),
            FaceDetect(resize),
            #Resize(resize, Image.BILINEAR),
            RandomErasing(p=1, scale=(0.025,0.025), ratio=(0.5,1)),
            Normalize(mean=mean, std=std)
        ])

    def __call__(self, image):
        return self.transform(image)


class FaceDetect(object):

    def __init__(self, resize=(380,380)):
        self.tf = transforms.ToPILImage()
        self.toTensor = transforms.ToTensor()
        self.center_crop = transforms.CenterCrop((380, 380))
        self.detector = MTCNN(image_size=resize[0], margin=150, post_process=False)

    def __call__(self, tensor):
        img = self.tf(tensor)
        face = self.detector(img) # tensor

        if face == None:
            face = self.center_crop(tensor) # tensor
        else:
            face = PIL.ImageOps.invert(self.tf(face))  #tensor -> pil
            face = self.toTensor(face)  # pil -> tensor
        return face

class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class Age11Labels(int, Enum):

    @staticmethod
    def labeling(value : int) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"age value shoule be numeric, {value}")

        intLable = min(value // 10, 10) # 10보다 작은 값            
        
        return intLable
	
class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []
    class_labels = []
	
    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def get_sampler(self) :
        class_sample_count = np.array([len(np.where(self.class_labels == t)[0]) for t in np.unique(self.class_labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in self.class_labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_weight, _ = random_split(samples_weight, [n_train, n_val], generator=torch.Generator().manual_seed(42))
        train_sampler = WeightedRandomSampler(train_weight, len(train_weight))
        return train_sampler

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                if (27<=age<=29) or (57<=age<=59) :
                    continue
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)
                
                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)
                self.class_labels.append(self.encode_multi_class(mask_label, gender_label, age_label))


    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        if index in self.indices["train"] :
            _transform =  self.transform
        else :
            _transform =  BaseAugmentation()
        return _transform(image), multi_class_label
					 
    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val], generator=torch.Generator().manual_seed(42))
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    if (27<=int(age)<=29) or (57<=int(age)<=59) :
                        continue
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset]:
																				  
        return [Subset(self, indices) for phase, indices in self.indices.items()]
    
    def get_sampler(self, phase) :
        _multi_class = []
        for _idx in self.indices[phase]:
            _temp = self.encode_multi_class(self.mask_labels[_idx],
                                    self.gender_labels[_idx],
                                    self.age_labels[_idx])
            _multi_class.append(_temp)
       
        class_sample_count = np.array([len(np.where(_multi_class == t)[0]) for t in np.unique(_multi_class)])
														   
        weight = 1. / class_sample_count
								  
        samples_weight = np.array([weight[t] for t in _multi_class])
        samples_weight = torch.from_numpy(samples_weight)
												  
        samples_weight = samples_weight.double()
        #n_val = int(len(self) * self.val_ratio)
        #n_train = len(self) - n_val
        #train_weight, _ = random_split(samples_weight, [n_train, n_val], generator=torch.Generator().manual_seed(42))
        train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return train_sampler

class CutMixDataset(MaskSplitByProfileDataset):
    num_classes = 2 # Gender class
    def __init__(self, data_dir, mean=(0.527, 0.465, 0.435), std=(0.24, 0.242, 0.243), val_ratio=0.2):
        
        self.age11_labels = []
        self.class_idx = [[] for i in range(0, 18)]  #클래스별 index
        
        self.istrain = []

        super().__init__(data_dir, mean, std, val_ratio)

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)
                    age11_label = Age11Labels.labeling(age)

                    #27 ~ 29, 57~59 데이터 제거
                    if (27<=int(age)<=29) or (57<=int(age)<=59) :
                        continue

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.age11_labels.append(age11_label)

                    multi_class = self.encode_multi_class(mask_label,gender_label,age_label)

                    self.class_idx[multi_class].append(cnt)

                    self.istrain.append(phase)

                    self.indices[phase].append(cnt)
                    cnt += 1


    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        _image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        
        #3,4,6 데이터 argumnetation
        age11_label = self.age11_labels[index]    
        
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        if self.istrain[index] == "train" and random.choice([True,False]):
            _idx2 = self.getRandom(multi_class_label)
            _image2 = self.read_image(_idx2)
            _image = self.transform(_image)
            _image2 = self.transform(_image2)

            image = cutmix(_image,_image2)
        else :           
            image = BaseAugmentation()(_image)

        if age11_label in [3,4,6,7]:
            image = RandomHorizontalFlip(p = 0.5)(image)
    
        
        return image, gender_label
        
    def getRandom(self, class_label):
        label = class_label
        #print(f"in random :{label}")
        return random.choice(self.class_idx[label]) #label 별 random idx

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]

class Age11Dataset(CutMixDataset):
    def __init__(self, data_dir, mean=(0.527, 0.465, 0.435), std=(0.24, 0.242, 0.243), val_ratio=0.2):
        super().__init__(data_dir, mean, std, val_ratio)

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        age11_label = self.age11_labels[index]
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        if index in self.indices["train"] :
            _transform =  self.transform
            image = _transform(image)
            if age11_label >= 6:
                image = RandomHorizontalFlip(p = 0.5)(image)
        else :
            _transform =  BaseAugmentation()
            image = _transform(image)

        return image, multi_class_label

class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.527, 0.465, 0.435), std=(0.24, 0.242, 0.243)):
        self.img_paths = img_paths
        self.transform = BaseAugmentation()

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)