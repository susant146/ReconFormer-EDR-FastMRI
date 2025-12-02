import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import h5py
import torch
import os
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# from models.recon_Update import train_recon
# from models.evaluation import evaluate_recon
from data.subsample import create_mask_for_mask_type
# from tensorboardX import SummaryWriter
# from utils.options import args_parser
# from models.evaluation import test_recon_save
from torch.utils.data import Subset
from data.mri_data import SliceData
from data import transforms
from data import transformsPB as Tpb
from data.subsample import create_mask_for_mask_type
from models.Recurrent_Transformer_new import ReconFormer_EDR
import pathlib
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
import fastmri
from data.combined_GradLoss import DualStreamLoss
from torch.nn import functional as F
# from fastmri.data import transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, resolution, which_challenge, mask_func=None, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(
                f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, mask, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            mask (numpy.array): Mask from the test dataset
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
        """
        target = Tpb.to_tensor(target.astype(complex)) # (Height x Width x 2)
        kspace = Tpb.fft2c_new(target)                 # (Height x Width x 2)
        
        # Apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = transforms.apply_mask(
                kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # Inverse Fourier Transform to get zero filled solution
        image = Tpb.ifft2c_new(masked_kspace) # (Height, Width, 2)

        # Absolute value
        abs_image = Tpb.complex_abs(image) # (height, width)
        mean = torch.tensor(0.0)
        std = abs_image.mean()
        # Normalize input
        image = image.permute(2, 0, 1)    # (2 x Height x Width)
        target = target.permute(2, 0, 1)  # (2 x Height x Width)
        image = transforms.normalize(image, mean, std, eps=0)
        masked_kspace = masked_kspace.permute(2, 0, 1)  # (2 x Height x Width)
        masked_kspace = transforms.normalize(masked_kspace, mean, std, eps=0)
        # Normalize target
        target = transforms.normalize(target, mean, std, eps=0)
        mask = mask.repeat(image.shape[1], 1, 1).squeeze().unsqueeze(0)  # (1 x Height x Width)
        return image, target, mean, std, attrs['norm'].astype(np.float32), fname, slice, attrs['max'].astype(np.float32), mask, masked_kspace

def _create_dataset(data_paths, transform, batch_size, shuffle, sample_rate):
    datasets = [SliceData(root=pathlib.Path(path), transform=transform, sample_rate=sample_rate, challenge='multicoil', sequence='PD') for path in data_paths] # , sequence='PD'
    # Step 2: Concatenate them into one dataset
    full_dataset = torch.utils.data.ConcatDataset(datasets)

    # Step 3: Compute 75% size and randomly shuffle indices
    total_size = len(full_dataset)
    train_size = int(0.50 * total_size)
    indices = torch.randperm(total_size).tolist()

    # Step 4: Create subset dataset
    subset_dataset = Subset(full_dataset, indices[:train_size])
    return DataLoader(subset_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)

def compute_ssim(output, target):
    output_np, target_np = output.squeeze().cpu().numpy(), target.squeeze().cpu().numpy()
    return ssim(output_np, target_np, data_range=target_np.max() - target_np.min())

def train_epoch(model, train_loader, optimizer):
    model.train()
    total_loss, start_time = 0, time.perf_counter()
    print('Length of Train Dataset: ', len(train_loader))
    for batch in train_loader:
        input, target, _, _, _, _, _, _, mask, masked_kspace = batch
        target = target.to(device).float()
        output = model(input.to(device).float(), masked_kspace.to(device).float(), mask.to(device))
        loss = F.l1_loss(Tpb.complex_abs(output.permute(0,2,3,1)), Tpb.complex_abs(target.permute(0,2,3,1)))
        # dual_stream_loss = DualStreamLoss(lambda_1=0.65, lambda_2=0.35, lambda_3=0.00)# lambda_1 [0.6] : L1_loss, Lambda_2 [0.4]:Grad Loss & lambda_3: SSIM Loss
        # loss = dual_stream_loss(Tpb.complex_abs(output.permute(0,2,3,1)).unsqueeze(0), Tpb.complex_abs(target.permute(0,2,3,1)).unsqueeze(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader), time.perf_counter() - start_time

def validate_epoch(model, val_loader):
    model.eval()
    total_loss, total_ssim, start_time = 0, 0, time.perf_counter()
    print('Length of Val Dataset: ', len(val_loader))
    with torch.no_grad():
        for batch in val_loader:
            input, target, _, _, _, _, _, _, mask, masked_kspace = batch
            target = target.to(device).float()
            output = model(input.to(device).float(), masked_kspace.to(device).float(), mask.to(device))
            loss = F.l1_loss(Tpb.complex_abs(output.permute(0,2,3,1)), Tpb.complex_abs(target.permute(0,2,3,1)))
            # dual_stream_loss = DualStreamLoss(lambda_1=0.65, lambda_2=0.35, lambda_3=0.00) # lambda_1 : L1_loss, Lambda_2:Grad Loss & lambda_3: SSIM Loss
            # loss = dual_stream_loss(Tpb.complex_abs(output.permute(0,2,3,1)).unsqueeze(0), Tpb.complex_abs(target.permute(0,2,3,1)).unsqueeze(0))
            total_loss += loss.item()
            total_ssim += compute_ssim(Tpb.complex_abs(output.permute(0,2,3,1)), Tpb.complex_abs(target.permute(0,2,3,1)))
    return total_loss / len(val_loader), total_ssim / len(val_loader), time.perf_counter() - start_time

def main():
    print(device)
    mask = create_mask_for_mask_type('random', [0.08], [4])
    bs = 1
    transform = DataTransform(320, which_challenge='multicoil', mask_func=mask, use_seed=False)
    train_data_paths = ["/home/susant/Knee_Multicoil_train_batch0/multicoil_train"]
    val_data_path = "/storage/FastMRI/Knee/knee_multicoil_val_subset"
    train_loader = _create_dataset(train_data_paths, transform, bs, True, 1.0)
    val_loader = _create_dataset([val_data_path], transform, bs, False, 1.0)
    
    model = ReconFormer_EDR(in_channels=2, out_channels=2, num_ch=(96, 48, 24), num_iter=4,
                         down_scales=(2, 1, 1.5), img_size=320, num_heads=(6, 6, 6), depths=(2, 1, 1),
                         window_sizes=(8, 8, 8), mlp_ratio=2.0, resi_connection='1conv',
                         use_checkpoint=(False, False, True, True, False, False)).to(device)
    
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    save_dir = "/home/susant/ReconFormer_MRI_recon/ReconFormer_savedModels"
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "ReconFormer_Acc4_log.txt")
    
    best_ssim, max_epochs, save_interval = 0.0, 50, 10

    with open(log_file, "w") as f:
        f.write("Epoch\tTrain_Loss\tVal_Loss\tVal_SSIM\tLR\tTrain_time\n")
    
    for epoch in range(max_epochs):
        train_loss, train_time = train_epoch(model, train_loader, optimizer)
        val_loss, val_ssim, val_time = validate_epoch(model, val_loader)
        scheduler.step()
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val SSIM = {val_ssim:.4f}, Train Time = {train_time:.2f}s, Val Time = {val_time:.2f}s")

        with open(log_file, "a") as f:
            f.write(f"{epoch}\t{train_loss:.4f}\t{val_loss:.4f}\t{val_ssim:.4f}\t{optimizer.param_groups[0]['lr']:.6f}\t{train_time:.4f}\n")
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save(model.state_dict(), os.path.join(save_dir, "ReconFormer_Acc4_best.pth"))
        if epoch % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"ReconFormer_Acc4_epoch{epoch}.pth"))

if __name__ == "__main__":
    main()
