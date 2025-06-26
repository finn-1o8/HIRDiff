import torch
import scipy.io as sio
import glob
import numpy as np
import os


class CustomNPYFolderDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, image_size=128):
        super(CustomNPYFolderDataset, self).__init__()
        self.image_paths = sorted(glob.glob(os.path.join(dataroot, "*.npy")))
        if not self.image_paths:
            raise FileNotFoundError(f"No .npy files found in {dataroot}")
        self.image_size = image_size
        print(f"Found {len(self.image_paths)} images in {dataroot}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img_np = np.load(path).astype(np.float32)
        img_tensor = torch.from_numpy(img_np)

        # Scale from [0, 1] range to [-1, 1] range for diffusion model
        img_tensor = (img_tensor * 2.0) - 1.0

        return {
            'LQ': img_tensor,
            'GT': img_tensor.clone(),
            'path': path
        }


def load_data(opt):
    dataset_name = opt['dataset']['name'] if isinstance(opt['dataset'], dict) else opt.dataset.name
    print(f"Loading data for dataset: {dataset_name}")

    if dataset_name == 'STURM_S2_6BAND':
        dataroot = opt['dataset']['dataroot'] if isinstance(opt['dataset'], dict) else opt.dataset.dataroot
        image_size = opt['dataset']['image_size'] if isinstance(opt['dataset'], dict) else opt.dataset.image_size
        batch_size = opt['inference']['params'].get('batch_size', 1) if isinstance(opt['inference'], dict) else opt.inference.params.get('batch_size', 1)

        dataset = CustomNPYFolderDataset(dataroot=dataroot, image_size=image_size)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return data_loader

    elif dataset_name in ['Houston', 'WDC', 'Salinas']:
        dataroot_mat = opt['dataset']['dataroot'] if isinstance(opt['dataset'], dict) else opt.dataset.dataroot
        data = sio.loadmat(dataroot_mat)
        if 'input' in data:
            return data['input'], data['gt'], data.get('sigma')
        else:
            return data.get('HSI_LR'), data.get('HSI_HR'), data.get('sigma')
    else:
        raise NotImplementedError('Dataset %s not recognized' % dataset_name)
