import os, pathlib
import csv

from typing import Callable

import torch
import torchvision
from torch.utils import data
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from pl_bolts.datamodules import ImagenetDataModule
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization

import numpy as np
import pandas as pd
import PIL.Image
import albumentations as A
import cv2
from collections import Counter
import scipy.io

class BFMDataset(data.Dataset):
    def __init__(self, data_dir, img_list, labels, normalization_meta, imsize=None, dataset=None):
        self.data_dir=data_dir
        self.img_list=img_list
        self.labels=labels #[num_imgs, #nodes]
        self.dataset=dataset
        self.normalize_mean=normalization_meta['mean']
        self.normalize_std=normalization_meta['std']

        if imsize==128:
            self.resize=146
            self.imsize=128
        elif imsize==224:
            self.resize=256
            self.imsize=224

    def verify_imgfile(self):
        img_paths=[os.path.join(self.data_dir, img_fn) + '.jpg' for img_fn in self.img_list]
        for img_path in img_paths:
            assert os.path.exists(img_paths), f'image: {img_path} not found.'

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_fn=self.img_list[index]

        img_path = os.path.join(self.data_dir, img_fn) + '.jpg'
        im=PIL.Image.open(img_path)
        im=transforms.ToTensor()(im)
        im=self.crop_head(im,extend=0.1)
        im=transforms.Resize(self.resize)(im)

        if self.dataset=='training':
            im = transforms.RandomCrop(self.imsize)(im)
            im = transforms.RandomGrayscale(p=0.2)(im)
        else:
            im = transforms.CenterCrop(self.imsize)(im)

        im=transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)(im)

        img_index=int(img_fn) #image filename corresponds to labels numpy array's row
        label=torch.as_tensor(self.labels[img_index,:].copy(),dtype=torch.float32)

        return im, label

    def crop_head(self,im,extend=None):
        device=im.device
        corners=torch.stack([im[:,0,0],im[:,0,-1],im[:,-1,0],im[:,-1,-1]])
        background=torch.mode(corners,axis=0).values.to(device)
        background=torch.all(im.permute(1,2,0)==background,axis=2)

        w=torch.where(torch.all(background,axis=0)==False)[0]
        h=torch.where(torch.all(background,axis=1)==False)[0]
        x1,x2=w.min().item(),w.max().item()
        y1,y2=h.min().item(),h.max().item()

        if type(extend) is float:
            x_max,y_max=im.shape[1],im.shape[2]
            x_center=torch.true_divide(x1+x2,2)
            y_center=torch.true_divide(y1+y2,2)
            x_extend=torch.true_divide(x2-x1,2)*(1+extend)
            y_extend=torch.true_divide(y2-y1,2)*(1+extend)
            x1,x2=max(0,int(x_center-x_extend)),min(int(x_center+x_extend),x_max)
            y1,y2=max(0,int(y_center-y_extend)),min(int(y_center+y_extend),y_max)

        crop=im[:,y1:y2+1,x1:x2+1]

        return crop

class BFMDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, train_label_path, meta_dir, test_dir=None, test_label_path=None, imsize=None, batch_size=256, val_size=0.1, num_workers=16, subsample_dataset_fraction=None):
        super().__init__()

        assert os.path.exists(train_dir), f'path: {train_dir} not found.'
        self.train_dir = train_dir
        assert os.path.exists(train_label_path), f'path: {train_label_path} not found.'
        self.train_label_path = train_label_path

        channel_dir=os.path.join(meta_dir, 'channel_meta.npz')
        assert os.path.exists(channel_dir), f'path: {channel_dir} not found.'
        self.normalize_meta_npz=np.load(channel_dir)

        if test_dir is not None:
            assert os.path.exists(test_dir), f'path: {test_dir} not found.'
            self.test_dir = test_dir
            assert os.path.exists(test_label_path), f'path: {test_label_path} not found.'
            self.test_label_path=test_label_path

        self.imsize=imsize
        self.batch_size=batch_size
        self.val_size=val_size
        self.num_workers=num_workers
        self.subsample_dataset_fraction=subsample_dataset_fraction

    def setup(self, stage: str = None):

        if stage == 'fit' or stage is None:
            img_indices=sorted(os.listdir(self.train_dir))
            img_indices=[img_index.split('.')[0] for img_index in img_indices]

            print('Splitting train/validation data using random state',42)
            X_train, X_val, y_train, y_val = train_test_split(img_indices, img_indices, test_size=self.val_size, random_state=42)

            self.train_list = X_train
            self.val_list = X_val

            if self.subsample_dataset_fraction is not None:
                self.train_list = self.train_list[:int(len(self.train_list)*self.subsample_dataset_fraction)]
                self.val_list = self.val_list[:int(len(self.val_list)*self.subsample_dataset_fraction)]

            self.train_labels=np.load(self.train_label_path, mmap_mode='r')

        if stage == 'testing':
            img_indices=sorted(os.listdir(self.test_dir))
            img_indices=[img_index.split('.')[0] for img_index in img_indices]

            latents=np.load(test_label_path)
            keys=latents.files
            self.test_labels=np.hstack(tuple([latents[key] for key in keys]))

    def train_dataloader(self, distributed=True):
        train_data = BFMDataset(self.train_dir, self.train_list, self.train_labels, self.normalize_meta_npz, self.imsize, 'training')
        if distributed:
            sampler=DistributedSampler(train_data, shuffle=False)
        else:
            sampler=None

        print('initializing training dataloader...')
        return DataLoader(train_data, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)

    def val_dataloader(self, distributed=True):

        val_data = BFMDataset(self.train_dir, self.val_list, self.train_labels, self.normalize_meta_npz, self.imsize, 'validation')
        if distributed:
            sampler=DistributedSampler(val_data, shuffle=False)
        else:
            sampler=None

        print('initializing validation dataloader...')
        return DataLoader(val_data, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)

    def test_dataloader(self):
        raise NotImplementedError

class ImagenetDataModuleGray(ImagenetDataModule):
    def train_transform(self) -> Callable:

        preprocessing = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            imagenet_normalization(),
        ])

        return preprocessing

class VGGFace2Dataset(data.Dataset):
    def __init__(self, data_dir, bb_df, img_list, class_to_ind_map, is_small_imgsize=False, dataset='training', preprocessed=False, do_normalize=True, p_gray = 0.2):
        self.data_dir = data_dir
        self.img_list = img_list
        self.dataset = dataset
        self.preprocessed = preprocessed
        self.p_gray = p_gray

        if is_small_imgsize == False:
            self.imgsize=224
            self.resize=256
        else:
            self.imgsize=128
            self.resize=146

        if do_normalize:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.6068, 0.4517, 0.3800], std=[0.2492, 0.2173, 0.2082])
            ])
        else:
            self.transform = transforms.ToTensor()
        self.img_info = []

        for i_img, img_file in enumerate(self.img_list):
            name_id = img_file.strip()
            class_id = img_file.strip().split('/')[0]
            ind = class_to_ind_map[class_id]

            if self.preprocessed==False:
                self.img_info.append({'NAME_ID': name_id, 'INDEX':ind})
            elif self.preprocessed==True:
                self.img_info.append({'NAME_ID': name_id, 'CLASS_ID': class_id, 'INDEX':ind})

            if self.dataset=='training':
                if i_img % 500000 == 0:
                    print(f"processing {i_img} images for {self.dataset} data.")
            else:
                if i_img % 50000 == 0:
                    print(f"processing {i_img} images for {self.dataset} data.")

        self.img_info = pd.DataFrame(self.img_info)

        if self.preprocessed==False:
            self.img_info = self.img_info.merge(bb_df, on='NAME_ID', how='left')

    def verify_imgfile(self):
        for i_img, img_file in enumerate(self.img_list):
            img_path = os.path.join(self.data_dir, img_file) + '.jpg'
            assert os.path.exists(img_path), f'image: {img_path} not found.'

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        info = self.img_info.iloc[index]
        img_path = os.path.join(self.data_dir, info.NAME_ID) + '.jpg'
        img = PIL.Image.open(img_path)

        ind = info.INDEX
        name_id = info.NAME_ID
        if self.preprocessed==False:
            X,Y,W,H = info.X, info.Y, info.W, info.H
            bb = (X, Y, X+W, Y+H)
            img = img.crop(bb)

            img = torchvision.transforms.Resize(self.resize)(img)

        if self.dataset == 'training':
            img = torchvision.transforms.RandomCrop(self.imgsize)(img)
            img = torchvision.transforms.RandomGrayscale(p=self.p_gray)(img)
        else:
            img = torchvision.transforms.CenterCrop(self.imgsize)(img)

        img = self.transform(img)

        return img, ind

class VGGFace2DataModule(pl.LightningDataModule):
    def __init__(self, train_dir, test_dir, train_bb_path, test_bb_path, identity_meta_path, class_map_path=None, batch_size=256, val_size=0.1, is_small_imgsize=False, random_state=None, num_workers=16, preprocessed=False, subsample_dataset_fraction=None, do_normalize=True,  p_gray=0.2):
        super().__init__()

        assert os.path.exists(train_dir), f'path: {train_dir} not found.'
        self.train_dir = train_dir

        assert os.path.exists(test_dir), f'path: {test_dir} not found.'
        self.test_dir = test_dir

        assert os.path.exists(train_bb_path), f'path: {train_bb_path} not found.'
        self.train_bb_path = train_bb_path

        assert os.path.exists(test_bb_path), f'path: {test_bb_path} not found.'
        self.test_bb_path = test_bb_path

        assert os.path.exists(identity_meta_path), f'path: {identity_meta_path} not found.'
        self.identity_meta_path = identity_meta_path

        if class_map_path is not None:
            pathlib.Path(class_map_path).mkdir(parents=True,exist_ok=True)
            assert os.path.exists(class_map_path), f'path: {class_map_path} not found.'
        self.class_map_path=class_map_path

        self.batch_size=batch_size
        self.val_size=val_size
        self.is_small_imgsize=is_small_imgsize
        self.random_state=random_state
        self.num_workers=num_workers
        self.preprocessed=preprocessed
        self.subsample_dataset_fraction=subsample_dataset_fraction
        self.do_normalize=do_normalize
        self.p_gray=p_gray

        if self.do_normalize:
            self.normalization_func = transforms.Normalize(mean=[0.6068, 0.4517, 0.3800], std=[0.2492, 0.2173, 0.2082])
            self.inverse_normalization_func = transforms.Compose([
                transforms.Normalize(mean=[0,0,0],std=[1/0.2492, 1/0.2173, 1/0.2082]),
                transforms.Normalize(mean=[-0.6068, -0.4517, -0.3800], std=[1,1,1])])
        else:
            self.normalization_func = None
            self.inverse_normalization_func = None

    def setup(self, stage: str = None):

        print('Loading training data bounding box csv file...')
        self.train_bb_df = pd.read_csv(self.train_bb_path)

        print('Loading testing data bounding box csv file...')
        self.test_bb_df = pd.read_csv(self.test_bb_path)

        print('Loading identity meta information...')
        identity_meta = pd.read_csv(self.identity_meta_path)
        identity_meta = identity_meta.rename(columns={' Name': 'Name'})
        identity_meta = identity_meta.rename(columns={' Sample_Num': 'Sample_Num'})
        self.identity_meta = identity_meta

        if stage == 'fit' or stage is None:

            print('Splitting train/validation data using random state',42)
            X_train, X_val, y_train, y_val = train_test_split(self.train_bb_df.NAME_ID, self.train_bb_df.CLASS_ID,
                                                              stratify=self.train_bb_df.CLASS_ID,
                                                              test_size=self.val_size, random_state=42)

            self.train_list = X_train.to_numpy()
            self.val_list = X_val.to_numpy()

            if self.subsample_dataset_fraction is not None:
                self.train_list = self.train_list[:int(len(self.train_list)*self.subsample_dataset_fraction)]
                self.val_list = self.val_list[:int(len(self.val_list)*self.subsample_dataset_fraction)]

            # from img_info to unique ID

            train_class_ids = list(np.unique(self.train_bb_df.CLASS_ID))
            self.train_class_map = {class_id: ind for ind, class_id in enumerate(train_class_ids)}

            # save the map
            if self.class_map_path is not None:
                ind_to_class_map = {ind: class_id for class_id, ind in self.train_class_map.items()}
                with open (os.path.join(self.class_map_path,'train_class_map.csv'), 'w') as file:
                    writer = csv.writer(file)
                    for ind, class_id in ind_to_class_map.items():
                        writer.writerow([ind, class_id])

            # calculate class_weights
            train_class_list=pd.DataFrame(train_class_ids, columns=['Class_ID']) #class index is mapped based on train_class_ids's order
            count = self.identity_meta[['Class_ID', 'Sample_Num']]
            count=train_class_list.merge(count, on='Class_ID', how='left')

            N = np.sum(count.Sample_Num)
            count['weight'] = count.apply(lambda x: N/x.Sample_Num, axis=1)
            total_weights = np.sum(count.weight)
            count['weight'] = count.apply(lambda x: x.weight/total_weights, axis=1)

            self.class_weights=count.weight

        if stage == 'testing' or stage is None:

            self.test_list = self.test_bb_df.NAME_ID.to_numpy()
            test_class_ids = list(np.unique(self.test_bb_df.CLASS_ID))
            self.test_class_map = {class_id: ind for ind, class_id in enumerate(test_class_ids)}

            if self.class_map_path is not None:
                ind_to_class_map = {ind: class_id for class_id, ind in self.test_class_map.items()}
                with open (os.path.join(self.class_map_path, 'test_class_map.csv'), 'w') as file:
                    writer = csv.writer(file)
                    for ind, class_id in ind_to_class_map.items():
                        writer.writerow([ind, class_id])

    def make_weights_for_balanced_classes(self, img_info):
        class_ids = set(img_info.CLASS_ID.tolist())

        count = self.identity_meta[['Class_ID', 'Sample_Num']]
        count = count[count['Class_ID'].isin(class_ids)]

        N = np.sum(count.Sample_Num)

        count['weight'] = N/count['Sample_Num']
        count = count[['Class_ID', 'weight']]
        count = count.rename(columns={'Class_ID': 'CLASS_ID'})

        weights = img_info.merge(count, on='CLASS_ID', how='left').weight.tolist()

        return weights

    def train_dataloader(self):

        train_data = VGGFace2Dataset(self.train_dir, self.train_bb_df, self.train_list, self.train_class_map, is_small_imgsize=self.is_small_imgsize, dataset='training', preprocessed=self.preprocessed, do_normalize=self.do_normalize, p_gray = self.p_gray)
        weights = self.make_weights_for_balanced_classes(train_data.img_info)
        weights = torch.DoubleTensor(weights)
        assert train_data.__len__() == len(weights)

        print('initializing training weighted random sampler with seed',self.random_state)
        weighted_sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=torch.Generator().manual_seed(self.random_state))
        return DataLoader(train_data, batch_size=self.batch_size, sampler=weighted_sampler, num_workers=self.num_workers)

    def val_dataloader(self, distributed=True):

        val_data = VGGFace2Dataset(self.train_dir, self.train_bb_df, self.val_list, self.train_class_map, is_small_imgsize=self.is_small_imgsize, dataset='validation', preprocessed=self.preprocessed, do_normalize=self.do_normalize)
        if distributed:
            sampler=DistributedSampler(val_data, shuffle=False)
        else:
            sampler=None

        print('initializing validation sampler')
        return DataLoader(val_data, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)

    def test_dataloader(self):
        raise NotImplementedError

class BFMIdentityDataset(data.Dataset):
    def __init__(self, data_dir, im_list, label_list, imsize=224, dataset='training'):
        self.data_dir = data_dir
        self.im_list = im_list
        self.label_list = label_list
        self.dataset = dataset

        if imsize == 224:
            self.imsize=224
            self.resize=256
        elif imsize == 128:
            self.imsize=128
            self.resize=146

        self.normalization_func=transforms.Normalize(mean=[0.55537174, 0.50970546, 0.48330758],std=[0.28882495, 0.26824081, 0.26588868])

        self.im_info=pd.DataFrame({'class_id':self.label_list,'image_path':self.im_list})

    def verify_imgfile(self):
        for im_path in self.im_info['image_path']:
            assert os.path.exists(im_path), f'image: {im_path} not found.'

    def __len__(self):
        return len(self.im_info)

    def __getitem__(self, index):
        info=self.im_info.iloc[index]
        im_path=info.image_path
        label=info.class_id

        im=PIL.Image.open(im_path)
        im=transforms.ToTensor()(im)
        im=self.crop_head(im,extend=0.1)
        
        if self.dataset=='training':
            im=im.permute(1,2,0).numpy()
            im=self.transform(im)
            im=torch.tensor(im).permute(2,0,1)
        
        im=transforms.Resize(self.resize)(im)
        if self.dataset=='training':
            im = transforms.RandomCrop(self.imsize)(im)
        else:
            im = transforms.CenterCrop(self.imsize)(im)
        im=self.normalization_func(im)

        return im, int(label)
    
    def transform(self,im):
        
        transform = A.Compose([
            A.OneOf([
                A.ToGray(p=0.5),
                A.RandomBrightnessContrast(p=1),
                A.Emboss(p=0.5),
                A.GaussNoise(var_limit=(0,0.1),p=0.5),
                A.MultiplicativeNoise(multiplier=(0.7,1.3), per_channel=True, elementwise=True, always_apply=True, p=1)
                ], p=1),
            A.OneOf([
                A.CoarseDropout(),
                A.Cutout(),
                A.GridDropout(p=0.1),
                A.GridDistortion(p=1),
                A.HorizontalFlip(p=0.5),
            ],p=1),
            A.OneOf([
                A.RandomFog(p=0.5),
                A.GaussianBlur(p=0.5),
                A.Blur(blur_limit=3,p=0.5),
                A.GlassBlur(p=0.5),
                A.OpticalDistortion(),
                A.Sharpen()
            ],p=1),
        ], p=0.95)
        
        im=transform(image=im)
        
        return im['image']
    
    def crop_head(self,im,extend=None):
        corners=torch.stack([im[:,0,0],im[:,0,-1],im[:,-1,0],im[:,-1,-1]])
        background=torch.mode(corners,axis=0).values
        background=background.to(im.device)
        background=torch.all(im.permute(1,2,0)==background,axis=2)

        w=torch.where(torch.all(background,axis=0)==False)[0]
        h=torch.where(torch.all(background,axis=1)==False)[0]
        x1,x2=w.min().item(),w.max().item()
        y1,y2=h.min().item(),h.max().item()

        if type(extend) is float:
            x_max,y_max=im.shape[1],im.shape[2]
            x_center=torch.true_divide(x1+x2,2)
            y_center=torch.true_divide(y1+y2,2)
            x_extend=torch.true_divide(x2-x1,2)*(1+extend)
            y_extend=torch.true_divide(y2-y1,2)*(1+extend)
            x1,x2=max(0,int(x_center-x_extend)),min(int(x_center+x_extend),x_max)
            y1,y2=max(0,int(y_center-y_extend)),min(int(y_center+y_extend),y_max)

        crop=im[:,y1:y2+1,x1:x2+1]

        return crop

class BFMIdentityDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, batch_size=256, val_size=0.1, imsize=224, num_workers=16, subsample_dataset_fraction=None):
        super().__init__()
        assert os.path.exists(train_dir), f'path: {train_dir} not found.'
        self.train_dir=train_dir

#         assert os.path.exists(all_image_path), f'path: {all_image_path} not found.'
#         self.all_image_path = all_image_path

        self.batch_size=batch_size
        self.val_size=val_size
        self.imsize=imsize
        self.num_workers=num_workers
        self.subsample_dataset_fraction=subsample_dataset_fraction

    def setup(self, stage: str = None):
        N_per_individual=363
        class_ids=os.listdir(self.train_dir)
        self.all_images=[os.path.join(self.train_dir, class_id, str(i_image).zfill(3)+'.jpg') for class_id in class_ids for i_image in range(0,N_per_individual)]
        self.all_ids=[image.split('/')[-2] for image in self.all_images]

        if stage == 'fit' or stage is None:

            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.all_images, self.all_ids,
                                                                                  stratify=self.all_ids,
                                                                                  test_size=self.val_size, random_state=42)

            if self.subsample_dataset_fraction is not None:
                self.X_train = self.X_train[:int(len(self.X_train)*self.subsample_dataset_fraction)]
                self.y_train = self.y_train[:int(len(self.y_train)*self.subsample_dataset_fraction)]
                self.X_val = self.X_val[:int(len(self.X_val)*self.subsample_dataset_fraction)]
                self.y_val = self.y_val[:int(len(self.y_val)*self.subsample_dataset_fraction)]

            assert len(self.X_train)==len(self.y_train)
            assert len(self.X_val)==len(self.y_val)

        if stage == 'testing':
            raise NotImplementedError

    def train_dataloader(self, distributed=True):
        train_data = BFMIdentityDataset(data_dir=self.train_dir, im_list=self.X_train, label_list=self.y_train, imsize=self.imsize, dataset='training')
        if distributed:
            sampler=DistributedSampler(train_data, shuffle=False)
        else:
            sampler=None

        print('initializing training dataloader...')
        return DataLoader(train_data, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)

    def val_dataloader(self, distributed=True):
        val_data = BFMIdentityDataset(data_dir=self.train_dir, im_list=self.X_val, label_list=self.y_val, imsize=self.imsize, dataset='validation')
        if distributed:
            sampler=DistributedSampler(val_data, shuffle=False)
        else:
            sampler=None

        print('initializing training dataloader...')
        return DataLoader(val_data, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)

    def test_dataloader(self):
        raise NotImplementedError