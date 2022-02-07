import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchsummary import summary
from loss import *
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 100
NUM_WORKERS = 8
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMAGE_DIR = "Round2_data/train/"
TRAIN_MASK_DIR = "Round2_Labels/train/"
VAL_IMAGE_DIR = "Round2_data/val/"
VAL_MASK_DIR = "Round2_Labels/val/"
INFO_PATH = "class_dict.csv"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data,target) in enumerate(loop):
        data = data.to(device=device)
        target = target.float().to(device=device)
        #forward
        target = target.permute(0,3,1,2)

    #forward
        with torch.cuda.amp.autocast():
            predictions = model(data)

            loss = loss_fn(predictions,target)

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    model = UNET(in_channels=3,out_channels=8,features=[8,16,32,64]).to(device)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = FocalLoss(gamma=5,logits=True)
    optimizer = optim.Adam(model.parameters(),lr=lr)
    
    train_transform = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT,width=IMAGE_WIDTH),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT,width=IMAGE_WIDTH),
            ToTensorV2(),
        ]
    )
    train_loader, val_loader = get_loaders(
        TRAIN_IMAGE_DIR, 
        TRAIN_MASK_DIR,
        VAL_IMAGE_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        info_path = INFO_PATH,
        train_transform=train_transform,
        val_transform=val_transform)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        checkpoint = {
            "state_dict":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            }
        save_checkpoints(checkpoint,file_name="checkpoint_FCloss_g5.pth.tar")
        check_accuracy(val_loader, model, epoch=epoch, batch_size=BATCH_SIZE, device=device)
        val_loss(val_loader, model, loss_fn=loss_fn, epoch=epoch, device = device)
        save_predictions_as_imgs(val_loader,model,path="predictions",device=device)

if __name__ =="__main__":
    main()