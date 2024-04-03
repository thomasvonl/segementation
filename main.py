import logging

import torch

import src.unet as generator

import src.preprocessing as data_loader
from src.deeplab import DeeplabV1
from src.eval import evaluation
from src.preprocessing import preprocess
from src.training import train_fn
from src.utils import show_image_label


@torch.inference_mode()
def generate():
    training, validation = data_loader.train_val_loader()
    images, labels = next(iter(training))
    print(images.shape)
    show_image_label(images[0] * 0.5 + 0.5, labels[0])


def train():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    training, validation = data_loader.train_val_loader(batch_size=4)

    model = generator.UNet(n_channels=1)
    model = DeeplabV1()
    model.to(device=device)
    train_fn(
        model=model,
        train_loader=training,
        val_loader=validation,
        epochs=40,
        batch_size=4,
        learning_rate=1e-5,
        device=device,
    )


if __name__ == "__main__":
    # model_path = "outputs/model_5.pth"
    # evaluation(model_path)
    train()
    # preprocess()
