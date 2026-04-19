from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.vaihingen_dataset import *
from timm.optim.lookahead import Lookahead

from catalyst import utils

# training hparam
max_epoch = 104
ignore_index = len(CLASSES)  
train_batch_size = 8
val_batch_size = 8
lr = 6e-4   
weight_decay = 2.5e-4   
backbone_lr = 6e-5  
backbone_weight_decay = 2.5e-4  
accumulate_n = 1 
num_classes = len(CLASSES)  
classes = CLASSES  

weights_name = ""
weights_path = "".format(weights_name)
test_weights_name = ""
log_name = "".format(weights_name)
monitor = ''   
monitor_mode = 'max'  
save_top_k = 1    
save_last = False   
check_val_every_n_epoch = 1   
gpus = []     
strategy = 'auto'  
pretrained_ckpt_path = None    
resume_ckpt_path = None  




#  define the network
from geoseg.models. import Model
net = Model(n_class=num_classes)

# define the loss
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
use_aux_loss = False          # 打开高频去噪辅助任务


# define the dataloader
train_dataset = VaihingenDataset(data_root="data/Vaihingen/train", mode='train',
                                 mosaic_ratio=0.25, transform=train_aug)

val_dataset = VaihingenDataset(data_root="data/Vaihingen/test",transform=val_aug)
test_dataset = VaihingenDataset(data_root="data/Vaihingen/test",
                                transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)     
import re
def process_model_params(model, layerwise_params=None, base_lr=None, base_weight_decay=None):
 
    if layerwise_params is None:
        return model.parameters()

    param_groups = []
    matched_params = set()
    for pattern, group_params in layerwise_params.items():
        group = {
            "params": [],
            "lr": group_params.get("lr", base_lr),
            "weight_decay": group_params.get("weight_decay", base_weight_decay)
        }
        for name, param in model.named_parameters():
            if re.match(pattern, name) and param.requires_grad:
                group["params"].append(param)
                matched_params.add(name)
        if group["params"]:
            param_groups.append(group)

    
    other_params = [p for n, p in model.named_parameters() if n not in matched_params and p.requires_grad]
    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": base_lr,
            "weight_decay": base_weight_decay
        })
    return param_groups


# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}  
net_params = process_model_params(net, layerwise_params=layerwise_params,
                                  base_lr=lr, base_weight_decay=weight_decay)

base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)  
optimizer = Lookahead(base_optimizer)  
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)  
