import enum
import cv2 as cv
import torch
import numpy as np
import pandas as pd
from PIL import Image
from dataset import *
from torch.utils.data import Dataset, DataLoader
import torchvision
import cv2
def get_image(image_path,label_path):
    pass

def get_labels_info(info_path):
    info = pd.read_csv(info_path)
    # info has format: [['obstacles' 59 193 246]...]
    # info = info.to_numpy()
    class_names = np.array(info["name"])
    labels_values = np.array(info[["r","g","b"]])
    return class_names, labels_values

def convert_data(img, label, info_path):
    img = img/255.0
    class_names, labels_values = get_labels_info(info_path)
    sematic_maps = []
    for color in labels_values:
        same = np.equal(label, color)
        class_map = np.all(same,axis=-1)
        sematic_maps.append(class_map)
    semantic_map = np.array(np.stack(sematic_maps,axis=-1))
    return img, semantic_map

def save_checkpoints(state, file_name="checkpoint.pth.tar"):
    print("=========> saving checkpoint")
    torch.save(state,file_name)

def load_checkpoints(checkpoint, model):
    print("==========> loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    return model

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    info_path,
    num_workers=4,
    pin_memory=True,
):
    train_ds = RC_dataset(
        image_dir=train_dir,
        label_dir = train_maskdir,
        info_path= info_path,
        transform=train_transform
    )
    val_ds = RC_dataset(
        image_dir=val_dir,
        label_dir= val_maskdir,
        info_path = info_path,
        transform=val_transform
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader,val_loader

def check_accuracy(loader, model, epoch, batch_size = 8, device="cuda"):
    num_correct = 0
    num_pixels = 0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            y = y.permute(0,3,1,2)
            preds = torch.sigmoid(model(x))
            preds = (preds>0.5).float()
            num_correct += (preds==y).sum()
            num_pixels += torch.numel(preds)
    print(f"EPOCH: {epoch} Got {num_correct}/{num_pixels*batch_size} ----> accuracy = {num_correct/num_pixels*100:.2f}")
    model.train()

def one_hot_reverse(preds,info_path="class_dict.csv"):
    class_names, labels_values = get_labels_info(info_path)
    preds_np = np.array(preds,dtype=np.float32)
    img = np.argmax(preds_np,axis=1)
    img_color = labels_values[img.astype(int)]
    #img_color = torch.Tensor(img_color).permute(0,3,1,2)
    return img_color

def save_predictions_as_imgs(loader, model, path = "predictions",device = "cuda",info_path="class_dict.csv"):
    model.eval()
    
    for idx, (x,y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            #preds = (preds >0.5).float()
        preds = one_hot_reverse(preds)
        for i in range(len(preds)):
            #BGRpreds = cv2.cvtColor(preds[i],cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{path}/model_predict/pred_{idx}_{i}.png",preds[i])
        # torchvision.utils.save_image(y.unsqueeze(1),f"{path}/label/label_{idx}.png")
    model.train()

def val_loss(loader, model, loss_fn, epoch, device="cuda"):
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            y = y.float().to(device = device)
            y = y.permute(0,3,1,2)
            preds = model(x)
            loss = loss_fn(y, preds)
            loss_total+= loss
    print(f"total loss in val dataset at epoch {epoch}: {loss_total}")
    return loss_total


if __name__ =="__main__":
    info_path = "/home/gumiho/project/WASR_seg/WASR/class_dict.csv"
    img = np.array(Image.open("/home/gumiho/project/WASR_seg/WASR/train/0001.png").convert("RGB"))
    label = np.array(Image.open("/home/gumiho/project/WASR_seg/WASR/train_labels/0001m.png").convert("RGB"))
    se = convert_data(img,label,info_path)
    print(se[0].shape)
    print(se[1].shape)
    x = np.random.randint(3,size=[8,3,256,256])
    x = torch.Tensor(x)
    img = one_hot_reverse(x)
    print(img.shape)
    print(img)
    import cv2
    cv2.imwrite("predictions/label/2.png",img[0])
    #torchvision.utils.save_image(img[0],f"predictions/label/1.png")

