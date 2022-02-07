import torch 
import pandas as pd  # For filelist reading
import myPytorchDatasetClass  # Custom dataset class, inherited from torch.utils.data.dataset
import numpy as np

def iou(pred, target, n_classes = 12):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)

  # Ignore IoU for background class ("0")
  for cls in xrange(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(float(intersection) / float(max(union, 1)))
  return np.array(ious)


def evaluate_performance(net):
    # Dataloader for test data
    batch_size = 1  
    filelist_name_test = '/path/to/my/test/filelist.txt'
    data_root_test = '/path/to/my/data/'
    dset_test = myPytorchDatasetClass.CustomDataset(filelist_name_test, data_root_test)
    test_loader = torch.utils.data.DataLoader(dataset=dset_test,  
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    data_info = pd.read_csv(filelist_name_test, header=None)
    num_test_files = data_info.shape[0]  
    sample_size = num_test_files

    # Containers for results
    preds = Variable(torch.zeros((sample_size, 60, 36, 60)))
    gts = Variable(torch.zeros((sample_size, 60, 36, 60)))

    dataiter = iter(test_loader) 
    for i in xrange(sample_size):
        images, labels, filename = dataiter.next()
        images = Variable(images).cuda()
        labels = Variable(labels)
        gts[i:i+batch_size, :, :, :] = labels
        outputs = net(images)
        outputs = outputs.permute(0, 2, 3, 4, 1).contiguous()
        val, pred = torch.max(outputs, 4)
        preds[i:i+batch_size, :, :, :] = pred.cpu()
    acc = iou(preds, gts)
    return acc