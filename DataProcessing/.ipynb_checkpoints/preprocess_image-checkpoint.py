#!/usr/bin/env python

import sys
import os
import glob
import shutil
from os import walk

#server default train files directory
train_dir = r"/opt/ml/input/data/train/images/"

#preprocessd_dir
preprocessed_dir = r"/opt/ml/input/data/train/processed_train_images/"

#label mapping dictionary
nameLabel: dict = {
    "incorrect_mask" : "_1",
    "mask1" : "_2",
    "mask2" : "_3",
    "mask3" : "_4",
    "mask4" : "_5",
    "mask5" : "_6",
    "normal" : "_7",
}

def getId(path: str) -> str:
    '''
    directory string에서 id 값을 반환합니다
    :param path: id 값을 찾을 디렉토리 문자열
    :return id: id 값
    '''
    tokens = path.split("/")
    tokens.reverse()
    
    temp = tokens[0].split("_")
    id = temp[0]
    return id
    
def getFileName(path: str) -> str:
    '''
    directory string에서 파일의 이름(name)을 반환합니다
    
    :param path: name을 찾을 디렉토리 문자열
    :return name: 디렉토리에서 파일의 이름
    '''
    tokens = path.split("/")
    tokens.reverse()
    
    temp = tokens[0].split(".") #filename.jpg
    name = temp[0]
    
    return name

def moveAndRenameFile(train_dir: str = train_dir, preprocessed_dir: str = preprocessed_dir):
    '''
    train_dir의 이미지 파일들을 preprocessed_dir의 경로로 이동 후 이름을 변경합니다.
    :param train_dir : image 파일의 루트 경로
    :param preprocessed_dir : 이동시키고자 하고자 하는 경로
    '''
    
    files = 0;
    #create folder
    try:
        if not os.path.exists(preprocessed_dir):
            os.makedirs(preprocessed_dir)
    except OSErrpr:
        print("Error: creating directory" + preprocessed_dir)
    
    train_dir_list = glob.glob(train_dir + r"*", recursive =True)

    for dir in train_dir_list:
        id = getId(dir) # 000001

        image_list = glob.glob(dir + r"/*")

        for image in image_list:
            files += 1;
            name = getFileName(image) # mask1
            oldfilename = name + ".jpg" # mask1.jpg

            #move file to preprocessed directory
            oldfile = image                           #/opt/ml/input/data/train/images/{foldername}/mask1.jpg
            movefile = preprocessed_dir + oldfilename #/opt/ml/input/data/train/processed_train_images/mask1.jpg
            shutil.copyfile(oldfile, movefile) # move file

            #change name to new format("{processed_dir}/id_{label}")
            newfile = preprocessed_dir + id + nameLabel[name] + ".jpg" # /opt/ml/input/data/train/processed_train_images/id_{lable}.jpg

            os.rename(movefile, newfile)
    
    print("Move files and rename is Done")
    print("From : " + train_dir)
    print("To : " + preprocessed_dir)
    print(f"preprocessed files : {files}")

if __name__  == "__main__" :
    if len(sys.argv) == 1:
        moveAndRenameFile()
    if len(sys.argv) == 3:
        moveAndRenameFile(sys.argv[1], sys.argv[2])
