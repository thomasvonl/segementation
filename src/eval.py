import h5py
import torch
import torch.nn.functional as F

import src.unet as generator
from src.deeplab import DeeplabV1
from src.utils import show_image_label


def load_test():
    images = h5py.File('data/testing_images.h5', 'r').get('image')[...]
    labels = h5py.File('data/testing_labels.h5', 'r').get('label')[...]

    return images, labels


def evaluation(model_path: str):
    images, true_label = load_test()
    images = images.transpose((0, 3, 1, 2))
    true_label = true_label.transpose((0, 3, 1, 2))

    print(images.min())
    print(images.max())

    assert images.min() >= 0 and images.max() <= 1
    images = (images[:5] - 0.5) / 0.5
    model = generator.UNet(n_channels=1)
    model = DeeplabV1()
    model.load_state_dict(torch.load(model_path))

    model.eval()

    pred_label = model((torch.tensor(images).to(dtype=torch.float32)))
    if isinstance(model, DeeplabV1):
        pred_label = F.interpolate(pred_label, size=images.shape[2:], mode='bilinear', align_corners=True)
    pred_label = (F.sigmoid(pred_label) > 0.5)
    show_image_label(images[3], pred_label[3])

