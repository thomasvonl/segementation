import logging
import torch

import wandb
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.deeplab import DeeplabV1
from src.utils import dice_loss, evaluate, show_image_label


def train_fn(model,
             device,
             train_loader,
             val_loader,
             epochs: int = 5,
             batch_size: int = 1,
             learning_rate: float = 1e-5,
             amp: bool = True,
             weight_decay: float = 1e-8,
             momentum: float = 0.999,
             gradient_clipping: float = 1.0
             ):
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must', tags=["deep_lab_v1"])
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, amp=amp)
    )

    n_train = len(train_loader)
    n_val = len(val_loader)

    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Device:          {device.type}
            Mixed Precision: {amp}
        ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    loss_fn = nn.BCEWithLogitsLoss()
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if isinstance(model, DeeplabV1):
                        masks_pred = F.interpolate(masks_pred, size=images.shape[2:], mode='bilinear', align_corners=True)
                    loss = loss_fn(masks_pred.squeeze(1), true_masks.float().squeeze(1))
                    loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float().squeeze(1))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                if global_step % 50 == 0:

                    val_score = evaluate(model, val_loader, device, amp)
                    scheduler.step(val_score)

                    logging.info('Validation Dice score: {}'.format(val_score))
                    experiment.log({
                        'validation Dice': val_score,
                        'images': wandb.Image(images[0].detach().cpu()),
                        'masks': {
                            'true': wandb.Image(true_masks[0].float().detach().cpu()),
                            'pred': wandb.Image((F.sigmoid(masks_pred[0]) > 0.5).float().detach().cpu()),
                        },
                        'step': global_step,
                    })

    # images, labels = next(iter(val_loader))
    # model.eval()
    # mask_pred = model(images.to(device))
    # show_image_label(images[0].detach().cpu(), (F.sigmoid(mask_pred[0]) > 0.5).detach().cpu())
    torch.save(model.state_dict(), "outputs/model_5.pth")


