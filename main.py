import torch
import tqdm

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data,target) in enumerate(loop):
        data = data.to(device=device)
        target = target.float().unsqueeze(1).to(device=device)
        #forward

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
