# import os
# import torch
# from torch.utils.data import DataLoader
# from torch import optim
# from Src import models
# from utils.fit import Fit
# from utils import utils
# import data_loader

# if __name__ == "__main__":
#     args = utils.get_parse()
#     if not os.path.exists(args.save_weights_path):
#         os.mkdir(args.save_weights_path)
#     model = models.model_B(image_size=args.image_size, in_chans=8)
#     if len(args.device) > 1:
#         model = torch.nn.DataParallel(model.to('cuda:0'), device_ids=[0, 1], output_device=0)
#     else:
#         model = model.to(device=args.device[0])
#     total = sum([param.nelement() for param in model.parameters()])
#     print("Number of parameter: %.2f" % total)
#     pg = utils.get_params_groups(model, args.weight_decay)
#     optimizer = optim.AdamW(pg, lr=1e-4, weight_decay=args.weight_decay)
#     begin_epoch = 0
#     # if len(os.listdir(args.save_weights_path)) > 0:
#     #     checkpoint = torch.load(args.save_weights_path + 'weights1.pth')
#     #     model.load_state_dict(checkpoint['model_dict'])
#     #     optimizer.load_state_dict(checkpoint['optimizer_dict'])
#     #     begin_epoch = checkpoint['epoch']

#     train_data, val_data = data_loader.get_data_path()
#     dataloaders = DataLoader(data_loader.Dataset(train_data, (args.image_size, args.image_size)),
#                              batch_size=args.batch_size,
#                              shuffle=True, num_workers=4)
#     dataloaders_val = DataLoader(data_loader.Dataset(val_data, (args.image_size, args.image_size)),
#                                  batch_size=args.val_batch_size,
#                                  shuffle=False, num_workers=4)
#     Fit(
#         model,
#         args,
#         optimizer,
#         dataloaders,
#         dataloaders_val,
#     ).fit(begin_epoch)






import os
import torch
from torch.utils.data import DataLoader
from torch import optim
from Src import models1
from utils.fit1 import Fit
from utils import utils
import data_loader

if __name__ == "__main__":
    args = utils.get_parse()

    if not os.path.exists(args.save_weights_path):
        os.mkdir(args.save_weights_path)

    model = models1.model_B(image_size=args.image_size, in_chans=8)

    # 确保设备字符串正确格式化
    if isinstance(args.device, tuple):
        args.device = list(args.device)
    elif isinstance(args.device, str):
        args.device = [args.device]

    # 检查设备字符串并正确设置模型的设备
    if len(args.device) > 1:
        device_ids = [i for i, d in enumerate(args.device) if 'cuda' in d]
        model = torch.nn.DataParallel(model.to(args.device[0]), device_ids=device_ids)
    else:
        model = model.to(device=args.device[0])

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2f" % total)

    pg = utils.get_params_groups(model, args.weight_decay)
    optimizer = optim.AdamW(pg, lr=1e-4, weight_decay=args.weight_decay)

    begin_epoch = 0

    train_data, val_data = data_loader.get_data_path()
    dataloaders = DataLoader(data_loader.Dataset(train_data, (args.image_size, args.image_size)),
                             batch_size=args.batch_size,
                             shuffle=True, num_workers=4)
    dataloaders_val = DataLoader(data_loader.Dataset(val_data, (args.image_size, args.image_size)),
                                 batch_size=args.val_batch_size,
                                 shuffle=False, num_workers=4)

    # Add codebook paths
    codebook_paths = {
        'CT': 'codebook_weights_CT.npy',
        'FLAIR': 'codebook_weights_FLAIR.npy',
        'PET': 'codebook_weights_PET.npy',
        'T1': 'codebook_weights_T1.npy'
    }

    Fit(
        model,
        args,
        optimizer,
        dataloaders,
        dataloaders_val,
        codebook_paths
    ).fit(begin_epoch)



