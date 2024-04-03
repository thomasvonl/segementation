import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms.v2 as transforms

from src.preprocessing_routines import xray_correct_threshold, create_patches, remove_bkgd_patches, standardize, \
    write_h5


def preprocess():
    dataset = h5py.File('resources/Au_20nm_87kx_16e_Images.h5', 'r')['images']
    labels = h5py.File('resources/Au_20nm_87kx_16e_Labels.h5', 'r')['labels']

    img_list = []
    lbl_list = []
    for i in range(dataset.shape[0]):
        img = xray_correct_threshold(dataset[i, :, :], threshold=400)
        img_list.append(img)
        lbl_list.append(labels[i, :, :])
    img_patches, lbl_patches = create_patches(img_list, lbl_list, patch_size=512)
    img_patches, lbl_patches = remove_bkgd_patches(img_patches, lbl_patches, threshold=400)
    images = np.float32(np.expand_dims(img_patches, axis=3))
    images = standardize(images)
    labels = np.float32(np.expand_dims(lbl_patches, axis=3))

    write_h5(dataset=images[:-5], label='image', h5_name="resources/training_images.h5")
    write_h5(dataset=labels[:-5], label='label', h5_name="resources/training_labels.h5")
    write_h5(dataset=images[-5:], label='image', h5_name="resources/testing_images.h5")
    write_h5(dataset=labels[-5:], label='label', h5_name="resources/testing_labels.h5")


def load():
    images = h5py.File('resources/training_images.h5', 'r')['image']
    labels = h5py.File('resources/training_labels.h5', 'r')['label']

    return images, labels


class CustomDataSet(Dataset):
    def __init__(self, transform):
        self.transform = transform
        images, labels = load()
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.transform(self.images[idx], self.labels[idx])
        assert image.min() >= -1 and image.max() <= 1
        assert label.min() >= 0 and image.max() <= 1
        return image, label


def train_val_loader(batch_size=32, shuffle=True) -> (DataLoader, DataLoader):
    transform = transforms.Compose([transforms.RandomVerticalFlip(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(90),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, ], [0.5, ])])
    dataset = CustomDataSet(transform=transform)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    validation_split = .1
    split = int(np.floor(validation_split * dataset_size))
    if shuffle:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    return train_loader, validation_loader
