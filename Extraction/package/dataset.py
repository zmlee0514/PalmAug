import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data as data
from PIL import Image
import numpy as np

def buildDatasets(dataset_params, dataset, trainingTransform, testingTransform, shot=5, val_ratio=0.2, test_ratio=0.2, indices=False):
    if indices:
        training_class_indices, testing_class_indices = indices
    elif test_ratio == 0:
        training_class_indices = range(dataset_params[dataset][1])
        testing_class_indices = range(dataset_params[dataset][1])
    else:
        training_class_indices, testing_class_indices = data.random_split(
            range(dataset_params[dataset][1]), [int(dataset_params[dataset][1]*(1-test_ratio)), int(dataset_params[dataset][1]*test_ratio)])
    
    if(dataset == "Tongji"):
        num_val_samples = int(20*val_ratio)
        trainingDataset = TongjiFewShotDataset(dataset_params[dataset][0], training_class_indices, 20, "gallery", trainingTransform)
        validationDataset = TongjiFewShotDataset(dataset_params[dataset][0], training_class_indices, 20, "probe", testingTransform)
        galleryDataset = TongjiFewShotDataset(dataset_params[dataset][0], testing_class_indices, shot, "gallery", testingTransform)
        probeDataset = TongjiFewShotDataset(dataset_params[dataset][0], testing_class_indices, 20-shot, "probe", testingTransform)
    elif(dataset == "Tongji_JPG"):
        num_val_samples = int(20*val_ratio)
        trainingDataset = TongjiJPGDataset(dataset_params[dataset][0], training_class_indices, 20, "gallery", trainingTransform)
        validationDataset = TongjiJPGDataset(dataset_params[dataset][0], training_class_indices, 20, "probe", testingTransform)
        galleryDataset = TongjiJPGDataset(dataset_params[dataset][0], testing_class_indices, shot, "gallery", testingTransform)
        probeDataset = TongjiJPGDataset(dataset_params[dataset][0], testing_class_indices, 20-shot, "probe", testingTransform)
    elif(dataset == "PolyU"):
        num_val_samples = int(12*val_ratio)
        trainingDataset = PolyUFewShotDataset(dataset_params[dataset][0], training_class_indices, 12, "gallery", trainingTransform)
        validationDataset = PolyUFewShotDataset(dataset_params[dataset][0], training_class_indices, 12, "probe", testingTransform)
        galleryDataset = PolyUFewShotDataset(dataset_params[dataset][0], testing_class_indices, shot, "gallery", testingTransform)
        probeDataset = PolyUFewShotDataset(dataset_params[dataset][0], testing_class_indices, 12-shot, "probe", testingTransform)
    elif(dataset == "PolyU_JPG"):
        num_val_samples = int(12*val_ratio)
        trainingDataset = PolyUJPGDataset(dataset_params[dataset][0], training_class_indices, 12, "gallery", trainingTransform)
        validationDataset = PolyUJPGDataset(dataset_params[dataset][0], training_class_indices, 12, "probe", testingTransform)
        galleryDataset = PolyUJPGDataset(dataset_params[dataset][0], testing_class_indices, shot, "gallery", testingTransform)
        probeDataset = PolyUJPGDataset(dataset_params[dataset][0], testing_class_indices, 12-shot, "probe", testingTransform)
    elif(dataset == "MPD_h"):
        num_val_samples = int(20*val_ratio)
        trainingDataset = MPDFewShotSingleDataset(dataset_params[dataset][0], training_class_indices, "h", 20, "gallery", trainingTransform)
        validationDataset = MPDFewShotSingleDataset(dataset_params[dataset][0], training_class_indices, "h", 20, "probe", testingTransform)
        galleryDataset = MPDFewShotSingleDataset(dataset_params[dataset][0], testing_class_indices, "h", shot, "gallery", testingTransform)
        probeDataset = MPDFewShotSingleDataset(dataset_params[dataset][0], testing_class_indices, "h", 20-shot, "probe", testingTransform)
    elif(dataset == "MPD_m"):
        num_val_samples = int(20*val_ratio)
        trainingDataset = MPDFewShotSingleDataset(dataset_params[dataset][0], training_class_indices, "m", 20, "gallery", trainingTransform)
        validationDataset = MPDFewShotSingleDataset(dataset_params[dataset][0], training_class_indices, "m", 20, "probe", testingTransform)
        galleryDataset = MPDFewShotSingleDataset(dataset_params[dataset][0], testing_class_indices, "m", shot, "gallery", testingTransform)
        probeDataset = MPDFewShotSingleDataset(dataset_params[dataset][0], testing_class_indices, "m", 20-shot, "probe", testingTransform)
    # elif(dataset == "MPD"):
    #     trainingDataset = MPDTrainingDataset(dataset_params[dataset][0], training_class_indices, "hm", trainingTransform)
    #     galleryDataset = MPDFewShotDataset(dataset_params[dataset][0], testing_class_indices, shot, "gallery", testingTransform)
    #     probeDataset = MPDFewShotDataset(dataset_params[dataset][0], testing_class_indices, 40-shot, "probe", testingTransform)
    else:
        print(dataset)
        
    return trainingDataset, validationDataset, galleryDataset, probeDataset, training_class_indices, testing_class_indices

def buildDataloaders(trainingDataset, validationDataset, galleryDataset, probeDataset, batch_size_train = 55, batch_size_test = 128):
    trainingDataloader = DataLoader(trainingDataset, batch_size=batch_size_train, shuffle=True)
    validationDataloader = DataLoader(validationDataset, batch_size=batch_size_test, shuffle=False)
    galleryDataloader = DataLoader(galleryDataset, batch_size=batch_size_test, shuffle=False)
    probeDataloader = DataLoader(probeDataset, batch_size=batch_size_test, shuffle=False)
    return trainingDataloader, validationDataloader, galleryDataloader, probeDataloader

## Tongji dataset
# contain both session of specific indices
class TongjiTrainingDataset(Dataset):
    '''
    all images of selected indices 
    '''
    def __init__(self, root, indices, transforms):
        # 圖片所在的資料夾
        self.root = root
        # 需要的類別編號
        self.indices = indices
        self.transforms = transforms

        self.fnames = []
        self.labels = []
        for c in self.indices:
            for i in range(c*10, c*10+10):
                self.fnames.append(os.path.join(self.root, 'session1/{:05d}.tiff'.format(i+1)))
                self.fnames.append(os.path.join(self.root, 'session2/{:05d}.tiff'.format(i+1)))
                # 左右手視為不同的類別
                self.labels.append(c)
                self.labels.append(c)
        self.labels = torch.Tensor(self.labels).long()

    def __getitem__(self, idx):
        # 利用路徑讀取圖片
        img = Image.open(self.fnames[idx])
        # 將輸入的圖片轉換成符合預訓練模型的形式
        img = self.transforms(img)
        # 補足3個channel
        # img = img.repeat(3,1,1)
        # 圖片相對應的 label
        label = self.labels[idx]
        return img, label
    
    def __len__(self):
        return len(self.fnames)
# contain first session of all indices, and second session of not selected
class TongjiTuningDataset(Dataset):
    '''
    testing set include half of the select indices, and this is the remain(useless)
    '''
    def __init__(self, root, indices, transforms):
        self.root = root
        # 註冊的類別編號
        self.indices = indices
        self.transforms = transforms

        self.fnames = []
        self.labels = []
        for i in range(6000):
            c = int(i/10)
            self.fnames.append(os.path.join(self.root, 'session1/{:05d}.tiff'.format(i+1)))
            self.labels.append(c)
            if c not in self.indices:
                self.fnames.append(os.path.join(self.root, 'session2/{:05d}.tiff'.format(i+1)))
                self.labels.append(c)
        self.labels = torch.Tensor(self.labels).long()

    def __getitem__(self, idx):
        img = Image.open(self.fnames[idx])
        img = self.transforms(img)
        label = self.labels[idx]
        return img, label
    
    def __len__(self):
        return len(self.fnames)

# only contain one session
class TongjiTestingDataset(Dataset):
    '''
    half of the selected indices
    '''
    def __init__(self, root, indices, mode, transforms):
        # 圖片所在的資料夾
        if mode == "probe":
            self.root = os.path.join(root, "session2")
        else:
            self.root = os.path.join(root, "session1")
        # self.root = root
        # 需要的類別編號
        self.indices = indices
        self.transforms = transforms

        self.fnames = []
        self.labels = []
        for c in self.indices:
            for i in range(c*10, c*10+10):
                self.fnames.append(os.path.join(self.root, '{:05d}.tiff'.format(i+1)))
                # 左右手視為不同的類別
                self.labels.append(c)
        self.labels = torch.Tensor(self.labels).long()

    def __getitem__(self, idx):
        # 利用路徑讀取圖片
        img = Image.open(self.fnames[idx])
        # 將輸入的圖片轉換成符合預訓練模型的形式
        img = self.transforms(img)
        # 補足3個channel
        # img = img.repeat(3,1,1)
        # 圖片相對應的 label
        label = self.labels[idx]
        return img, label
    
    def __len__(self):
        return len(self.fnames)
    
# flexibly spliting support set and query set
class TongjiFewShotDataset(Dataset):
    '''
    mode == gallery, get num_samples start from 1 to 20 of each class
    mode == probe, get num_samples start from 20 to 1 of each class
    '''
    def __init__(self, root, indices, num_samples, mode, transforms):
        # 圖片所在的資料夾
        self.root = root
        # 需要的類別編號
        self.indices = indices
        self.transforms = transforms
        if num_samples > 20:
            raise BaseException("Number of samples larger than the limit")
        else:
            self.session1 = 10 if num_samples > 10 else num_samples
            self.session2 = num_samples - self.session1
            if mode == "probe":
                self.session1, self.session2 = self.session2, self.session1
                self.session1 = range(10-self.session1, 10)
                self.session2 = range(10-self.session2, 10)
            else:
                self.session1 = range(self.session1)
                self.session2 = range(self.session2)

        self.fnames = []
        self.labels = []
        for c in self.indices:
            # get images from session1
            for i in self.session1: # [0,1,2,3,4,5,6,7,8,9]
                self.fnames.append(os.path.join(self.root, 'session1/{:05d}.tiff'.format(c*10+i+1)))
                self.labels.append(c)
            # get images from session2
            for i in self.session2: # [0,1,2,3,4,5,6,7,8,9]
                self.fnames.append(os.path.join(self.root, 'session2/{:05d}.tiff'.format(c*10+i+1)))
                self.labels.append(c)
        self.labels = torch.Tensor(self.labels).long()

    def __getitem__(self, idx):
        # 利用路徑讀取圖片
        img = Image.open(self.fnames[idx])
        img = np.asarray(img)
        # 將輸入的圖片轉換成符合預訓練模型的形式
        img = self.transforms(img)
        # 補足3個channel
        # img = img.repeat(3,1,1)
        # 圖片相對應的 label
        label = self.labels[idx]
        return img, label
    
    def __len__(self):
        return len(self.fnames)
    
# flexibly spliting support set and query set
class TongjiRotationCopyDataset(Dataset):
    '''
    mode == gallery, get num_samples start from 1 to 20 of each class
    mode == probe, get num_samples start from 20 to 1 of each class
    '''
    def __init__(self, root, indices, num_samples, mode, transforms):
        # 圖片所在的資料夾
        self.root = root
        # 需要的類別編號
        self.indices = indices
        self.transforms = transforms
        self.num_samples = num_samples
        if num_samples > 20:
            raise BaseException("Number of samples larger than the limit")
        else:
            self.session1 = 10 if num_samples > 10 else num_samples
            self.session2 = num_samples - self.session1
            if mode == "probe":
                self.session1, self.session2 = self.session2, self.session1
                self.session1 = range(10-self.session1, 10)
                self.session2 = range(10-self.session2, 10)
            else:
                self.session1 = range(self.session1)
                self.session2 = range(self.session2)

        self.fnames = []
        self.labels = []
        for c in self.indices:
            # get images from session1
            for i in self.session1: # [0,1,2,3,4,5,6,7,8,9]
                self.fnames.append(os.path.join(self.root, 'session1/{:05d}.tiff'.format(c*10+i+1)))
                self.labels.append(c)
            # get images from session2
            for i in self.session2: # [0,1,2,3,4,5,6,7,8,9]
                self.fnames.append(os.path.join(self.root, 'session2/{:05d}.tiff'.format(c*10+i+1)))
                self.labels.append(c)
        self.labels = torch.Tensor(self.labels).long()

    def __getitem__(self, idx):
        i = idx % (len(self.indices) * self.num_samples)
        quotient = int(idx / (len(self.indices) * self.num_samples))
        img = Image.open(self.fnames[i])
        img = np.asarray(img)
        if quotient > 0:
            img = np.rot90(img, quotient, (0,1)) # will rotate 1,2,3 times
        img = self.transforms(img)
        label = self.labels[i]
        return img, label + 600 * quotient
    
    def __len__(self):
        return len(self.fnames)*4
    
## PolyU dataset
# contain both session of specific indices
class PolyUTrainingDataset(Dataset):
    '''
    all images of selected indices 
    '''
    def __init__(self, root, indices, transforms):
        # 圖片所在的資料夾
        self.root = root
        # 需要的類別編號
        self.indices = indices
        self.transforms = transforms

        self.fnames = [[],[],[]]  # R,G,B
        self.labels = []
        for c in self.indices:
            for i in range(6):
                self.fnames[0].append(os.path.join(self.root, 'Multispectral_R/{:03d}/1_{:02d}_s.bmp'.format(c+1, i+1)))
                self.fnames[1].append(os.path.join(self.root, 'Multispectral_G/{:03d}/1_{:02d}_s.bmp'.format(c+1, i+1)))
                self.fnames[2].append(os.path.join(self.root, 'Multispectral_B/{:03d}/1_{:02d}_s.bmp'.format(c+1, i+1)))
                self.fnames[0].append(os.path.join(self.root, 'Multispectral_R/{:03d}/2_{:02d}_s.bmp'.format(c+1, i+1)))
                self.fnames[1].append(os.path.join(self.root, 'Multispectral_G/{:03d}/2_{:02d}_s.bmp'.format(c+1, i+1)))
                self.fnames[2].append(os.path.join(self.root, 'Multispectral_B/{:03d}/2_{:02d}_s.bmp'.format(c+1, i+1)))
                # 2 sessions
                self.labels.append(c)
                self.labels.append(c)
        self.labels = torch.Tensor(self.labels).long()

    def __getitem__(self, idx):
        img_R = Image.open(self.fnames[0][idx])
        img_G = Image.open(self.fnames[1][idx])
        img_B = Image.open(self.fnames[2][idx])
        img = np.dstack((img_R,img_G,img_B))
        img = self.transforms(img)
        label = self.labels[idx]
        return img, label
    
    def __len__(self):
        return len(self.labels)
    
# only contain one session
class PolyUTestingDataset(Dataset):
    '''
    half of the selected indices
    '''
    def __init__(self, root, indices, mode, transforms):
        self.root = root
        # 需要的類別編號
        self.indices = indices
        self.transforms = transforms
        # 決定session
        if mode == "probe":
            self.session = 2
        else:
            self.session = 1

        self.fnames = [[],[],[]]  # R,G,B
        self.labels = []
        for c in self.indices:
            for i in range(6):
                self.fnames[0].append(os.path.join(self.root, 'Multispectral_R/{:03d}/{}_{:02d}_s.bmp'.format(c+1, self.session, i+1)))
                self.fnames[1].append(os.path.join(self.root, 'Multispectral_G/{:03d}/{}_{:02d}_s.bmp'.format(c+1, self.session, i+1)))
                self.fnames[2].append(os.path.join(self.root, 'Multispectral_B/{:03d}/{}_{:02d}_s.bmp'.format(c+1, self.session, i+1)))
                self.labels.append(c)
        self.labels = torch.Tensor(self.labels).long()

    def __getitem__(self, idx):
        img_R = Image.open(self.fnames[0][idx])
        img_G = Image.open(self.fnames[1][idx])
        img_B = Image.open(self.fnames[2][idx])
        img = np.dstack((img_R,img_G,img_B))
        img = self.transforms(img)
        label = self.labels[idx]
        return img, label
    
    def __len__(self):
        return len(self.labels)
    
# flexibly spliting support set and query set
class PolyUFewShotDataset(Dataset):
    '''
    mode == gallery, get num_samples start from 1 to 20 of each class
    mode == probe, get num_samples start from 20 to 1 of each class
    '''
    def __init__(self, root, indices, num_samples, mode, transforms):
        self.root = root
        self.indices = indices
        self.transforms = transforms
        if num_samples > 12:
            raise BaseException("Number of samples larger than the limit")
        else:
            session1 = 6 if num_samples > 6 else num_samples
            session2 = num_samples - session1
            if mode == "probe":
                session1, session2 = session2, session1
                self.session1 = range(6-session1, 6)
                self.session2 = range(6-session2, 6)
            else:
                self.session1 = range(session1)
                self.session2 = range(session2)

        self.fnames = []
        self.labels = []
        for c in self.indices:
            # get images from session1
            for i in self.session1:
                fname = []
                for channel in "RGB":
                    fname.append(os.path.join(self.root, 'Multispectral_{}/{:03d}/{}_{:02d}_s.bmp'.format(channel, c+1, 1, i+1)))
                self.fnames.append(fname)
                self.labels.append(c)
            # get images from session2
            for i in self.session2:
                fname = []
                for channel in "RGB":
                    fname.append(os.path.join(self.root, 'Multispectral_{}/{:03d}/{}_{:02d}_s.bmp'.format(channel, c+1, 2, i+1)))
                self.fnames.append(fname)
                self.labels.append(c)
        self.labels = torch.Tensor(self.labels).long()

    def __getitem__(self, idx):
        imgs = []
        for i in self.fnames[idx]:
            imgs.append(Image.open(i))
        img = np.dstack(imgs)
        img = self.transforms(img)
        label = self.labels[idx]
        return img, label
    
    def __len__(self):
        return len(self.fnames)
    
# flexibly spliting support set and query set
class PolyURotationCopyDataset(Dataset):
    '''
    mode == gallery, get num_samples start from 1 to 20 of each class
    mode == probe, get num_samples start from 20 to 1 of each class
    '''
    def __init__(self, root, indices, num_samples, mode, transforms):
        self.root = root
        self.indices = indices
        self.transforms = transforms
        self.num_samples = num_samples
        if num_samples > 12:
            raise BaseException("Number of samples larger than the limit")
        else:
            session1 = 6 if num_samples > 6 else num_samples
            session2 = num_samples - session1
            if mode == "probe":
                session1, session2 = session2, session1
                self.session1 = range(6-session1, 6)
                self.session2 = range(6-session2, 6)
            else:
                self.session1 = range(session1)
                self.session2 = range(session2)

        self.fnames = []
        self.labels = []
        for c in self.indices:
            # get images from session1
            for i in self.session1:
                fname = []
                for channel in "RGB":
                    fname.append(os.path.join(self.root, 'Multispectral_{}/{:03d}/{}_{:02d}_s.bmp'.format(channel, c+1, 1, i+1)))
                self.fnames.append(fname)
                self.labels.append(c)
            # get images from session2
            for i in self.session2:
                fname = []
                for channel in "RGB":
                    fname.append(os.path.join(self.root, 'Multispectral_{}/{:03d}/{}_{:02d}_s.bmp'.format(channel, c+1, 2, i+1)))
                self.fnames.append(fname)
                self.labels.append(c)
        self.labels = torch.Tensor(self.labels).long()

    def __getitem__(self, idx):
        imgs = []
        i = idx % (len(self.indices) * self.num_samples)
        quotient = int(idx / (len(self.indices) * self.num_samples))
        for path in self.fnames[i]:
            imgs.append(Image.open(path))
        img = np.dstack(imgs)
        if quotient > 0:
            img = np.rot90(img, quotient, (0,1)) # will rotate 1,2,3 times
        img = self.transforms(img)
        label = self.labels[i]
        return img, label + 500 * quotient
    
    def __len__(self):
        return len(self.fnames)*4
    
## MPD dataset
# contain all data of specific phone
class MPDTrainingDataset(Dataset):
    '''
    all images of selected indices 
    '''
    def __init__(self, root, indices, phone, transforms):
        self.root = root
        self.indices = indices
        self.phone = phone
        self.transforms = transforms

        self.fnames = []
        self.labels = []
        for c in self.indices:
            for i in range(10):
                for p in self.phone:
                    self.fnames.append(os.path.join(self.root, '{:03d}_{}_{}_{}_{:02d}_ROI.jpeg'.format(c+1, 1, p, "l", i+1))) # 左手session1
                    self.labels.append(2*c)
                    self.fnames.append(os.path.join(self.root, '{:03d}_{}_{}_{}_{:02d}_ROI.jpeg'.format(c+1, 2, p, "l", i+1))) # 左手session2
                    self.labels.append(2*c)
                    self.fnames.append(os.path.join(self.root, '{:03d}_{}_{}_{}_{:02d}_ROI.jpeg'.format(c+1, 1, p, "r", i+1))) # 右手session1
                    self.labels.append(2*c+1)
                    self.fnames.append(os.path.join(self.root, '{:03d}_{}_{}_{}_{:02d}_ROI.jpeg'.format(c+1, 2, p, "r", i+1))) # 右手session2
                    self.labels.append(2*c+1)
        self.labels = torch.Tensor(self.labels).long()

    def __getitem__(self, idx):
        img = Image.open(self.fnames[idx])
        img = self.transforms(img)
        label = self.labels[idx]
        return img, label
    
    def __len__(self):
        return len(self.fnames)

# flexibly spliting support set and query set
class MPDFewShotDataset(Dataset):
    '''
    can only use for single phone
    mode == gallery, get num_samples start from 1 to 20 of each class
    mode == probe, get num_samples start from 20 to 1 of each class
    '''
    def __init__(self, root, indices, num_samples, mode, transforms):
        self.root = root
        self.indices = indices
        self.transforms = transforms
        self.num_samples = num_samples
        if num_samples > 40:
            raise BaseException("Number of samples larger than the limit")
        else:
            if mode == "probe": # count backward
                self.session = [2,1]
                self.phone = "mh"
                self.samples = range(9, -1, -1)
            else:
                self.session = [1,2]
                self.phone = "hm"
                self.samples = range(10)

        
        self.fnames = []
        self.labels = []
        for c in self.indices:
            count = 0
            for s in self.session:
                for i in self.samples:
                    for p in self.phone:
                        if count >= self.num_samples:
                            break
                        self.fnames.append(os.path.join(self.root, '{:03d}_{}_{}_{}_{:02d}_ROI.jpeg'.format(c+1, s, p, "l", i+1))) # 左手session1
                        self.labels.append(2*c)
                        self.fnames.append(os.path.join(self.root, '{:03d}_{}_{}_{}_{:02d}_ROI.jpeg'.format(c+1, s, p, "r", i+1))) # 右手session1
                        self.labels.append(2*c+1)
                        count += 1
        self.labels = torch.Tensor(self.labels).long()

    def __getitem__(self, idx):
        img = Image.open(self.fnames[idx])
        img = self.transforms(img)
        label = self.labels[idx]
        return img, label
    
    def __len__(self):
        return len(self.fnames)
    
class MPDFewShotSingleDataset(Dataset):
    '''
    can only use for single phone
    mode == gallery, get num_samples start from 1 to 20 of each class
    mode == probe, get num_samples start from 20 to 1 of each class
    '''
    def __init__(self, root, indices, phone, num_samples, mode, transforms):
        self.root = root
        self.indices = indices
        self.phone = phone
        self.transforms = transforms
        self.num_samples = num_samples
        if num_samples > 20:
            raise BaseException("Number of samples larger than the limit")
        else:
            session1 = 10 if num_samples > 10 else num_samples
            session2 = num_samples - session1
            if mode == "probe": # count backward
                session1, session2 = session2, session1
                self.session1 = range(10-session1, 10)
                self.session2 = range(10-session2, 10)
            else:
                self.session1 = range(session1)
                self.session2 = range(session2)

        
        self.fnames = []
        self.labels = []
        for c in self.indices:
            # get images from session1
            for i in self.session1: # [0,1,2,3,4,5,6,7,8,9]
                self.fnames.append(os.path.join(self.root, '{:03d}_{}_{}_{}_{:02d}_ROI.jpeg'.format(c+1, 1, self.phone, "l", i+1))) # 左手session1
                self.labels.append(2*c)
                self.fnames.append(os.path.join(self.root, '{:03d}_{}_{}_{}_{:02d}_ROI.jpeg'.format(c+1, 1, self.phone, "r", i+1))) # 右手session1
                self.labels.append(2*c+1)
            # get images from session2
            for i in self.session2: # [0,1,2,3,4,5,6,7,8,9]
                self.fnames.append(os.path.join(self.root, '{:03d}_{}_{}_{}_{:02d}_ROI.jpeg'.format(c+1, 2, self.phone, "l", i+1))) # 左手session2
                self.labels.append(2*c)
                self.fnames.append(os.path.join(self.root, '{:03d}_{}_{}_{}_{:02d}_ROI.jpeg'.format(c+1, 2, self.phone, "r", i+1))) # 右手session2
                self.labels.append(2*c+1)
        self.labels = torch.Tensor(self.labels).long()

    def __getitem__(self, idx):
        img = Image.open(self.fnames[idx])
        img = np.asarray(img)
        img = self.transforms(img)
        label = self.labels[idx]
        return img, label
    
    def __len__(self):
        return len(self.fnames)
    
class MPDFewShotSingleCopyDataset(Dataset):
    '''
    can only use for single phone
    mode == gallery, get num_samples start from 1 to 20 of each class
    mode == probe, get num_samples start from 20 to 1 of each class
    '''
    def __init__(self, root, indices, phone, num_samples, mode, transforms):
        self.root = root
        self.indices = indices
        self.phone = phone
        self.transforms = transforms
        self.num_samples = num_samples
        if num_samples > 20:
            raise BaseException("Number of samples larger than the limit")
        else:
            session1 = 10 if num_samples > 10 else num_samples
            session2 = num_samples - session1
            if mode == "probe": # count backward
                session1, session2 = session2, session1
                self.session1 = range(10-session1, 10)
                self.session2 = range(10-session2, 10)
            else:
                self.session1 = range(session1)
                self.session2 = range(session2)

        
        self.fnames = []
        self.labels = []
        for c in self.indices:
            # get images from session1
            for i in self.session1: # [0,1,2,3,4,5,6,7,8,9]
                self.fnames.append(os.path.join(self.root, '{:03d}_{}_{}_{}_{:02d}_ROI.jpeg'.format(c+1, 1, self.phone, "l", i+1))) # 左手session1
                self.labels.append(2*c)
                self.fnames.append(os.path.join(self.root, '{:03d}_{}_{}_{}_{:02d}_ROI.jpeg'.format(c+1, 1, self.phone, "r", i+1))) # 右手session1
                self.labels.append(2*c+1)
            # get images from session2
            for i in self.session2: # [0,1,2,3,4,5,6,7,8,9]
                self.fnames.append(os.path.join(self.root, '{:03d}_{}_{}_{}_{:02d}_ROI.jpeg'.format(c+1, 2, self.phone, "l", i+1))) # 左手session2
                self.labels.append(2*c)
                self.fnames.append(os.path.join(self.root, '{:03d}_{}_{}_{}_{:02d}_ROI.jpeg'.format(c+1, 2, self.phone, "r", i+1))) # 右手session2
                self.labels.append(2*c+1)
        self.labels = torch.Tensor(self.labels).long()

    def __getitem__(self, idx):    
        i = idx % (len(self.indices)*2 * self.num_samples)
        quotient = int(idx / (len(self.indices)*2 * self.num_samples))
        img = Image.open(self.fnames[i])
        img = np.asarray(img)
        if quotient > 0:
            img = np.rot90(img, quotient, (0,1)) # will rotate 1,2,3 times
        img = self.transforms(img)
        label = self.labels[i]
        return img, label + 400 * quotient
    
    def __len__(self):
        return len(self.fnames)*4