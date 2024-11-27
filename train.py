import time
from pathlib import Path
import os
from torch.nn import DataParallel

from src.configs import TrainConfig
import numpy as np
from tqdm import tqdm
import torch
import sys
import GPUtil
from tabulate import tabulate

from multiprocessing import cpu_count
import matplotlib.pyplot as plt
from datetime import datetime


from src.dataset import getDataLoader
from src.utils.deep import NetPhase
from src.cycle_gan_model import CycleGANModel
from src.cut_model import CUTModel
from src.test import get_fid_score
from torch.utils.tensorboard import SummaryWriter
torch.cuda.empty_cache()


config = TrainConfig()

####

train_loader = getDataLoader(is_train=True, bandwidth=config.wl, batch_size=config.data_loader.batch_size,
                             path_to_data='data')  # create a Dataset given opt.dataset_mode and other options
val_loader = getDataLoader(is_train=False, bandwidth=config.wl, batch_size=config.data_loader.batch_size,
                           path_to_data='data')  # create a Dataset for evaluating the results after each iteration
print(f"The number of training images = {len(train_loader) * config.data_loader.batch_size}, val images = {len(val_loader) * config.data_loader.batch_size}")

####

# config.model = "CUT" # "CycleGan"
# config.thermal.is_fpa_input = True
# config.thermal.is_physical_model = True
gpus = GPUtil.getGPUs()



backbone = CUTModel if config.model == "CUT" else CycleGANModel
model = backbone(config)

model.setup(config)



####

# TODO: update order of visualized images (mono_phys before mono_fake)
path_to_save = Path.cwd() / 'results' / 'train' / datetime.now().strftime("%Y%m%d_h%Hm%Ms%S")
images_path = path_to_save / 'images'
path_to_save.mkdir(parents=True, exist_ok=True)
images_path.mkdir(parents=True, exist_ok=True)

tensorboard = SummaryWriter(log_dir=path_to_save / 'experiment')
tot_epochs = config.scheduler.n_epochs + config.scheduler.n_epochs_decay + 1
best_fid_score = np.inf  # initialize fid threshold for best solution saving
rand_val_idx = np.random.randint(low=0, high=len(val_loader))  # used to randomly pick image for saving
for epoch in range(1, tot_epochs):
    # Train
    if not isinstance(NetPhase.train, NetPhase):
        raise TypeError(f"Expected phase to be an instance of NetPhase, got {type(NetPhase.train)}")
    model.set_phase(NetPhase.train)

    # if torch.cuda.device_count() > 1:
    #     model = DataParallel(model)
    # model.to('cuda')
    for i, data in enumerate(tqdm(train_loader, postfix="Train", desc=f"Epoch {epoch}|{tot_epochs-1}")):
        if epoch == 1 and i == 0 and 'cut' in str(model.__class__).lower():  # first iteration:
            model.data_dependent_initialize(data)
        model.set_input(data)
        model.forward()

        if config.network.gan_mode == "wgangp" and i % config.network.n_critic:
            train_gen = False
        else:
            train_gen = True
        model.optimize_parameters(train_gen)
    model.update_loss(epoch, len(train_loader))

    # Validate
    model.set_phase(NetPhase.val)

    with torch.inference_mode():
        for i, data in enumerate(tqdm(val_loader, postfix="Validate", desc=f"Epoch {epoch}|{tot_epochs-1}")):
            t_iter_start = time.time()
            model.set_input(data)
            model.forward()

            # additionally calculate losses for visualization purposes:
            model.calc_loss_D()
            model.calc_loss_G()
            model.update_agg_loss()

            model.print_current_losses(epoch=epoch, iters=i, t_iter_start=t_iter_start)
            model.plot_losses(tensorboard, epoch)

            # track visual performance:
            if i in [0, rand_val_idx]:
                plt.figure(figsize=(20, 10))
                plt.imshow(model.gen_vis_grid().permute(1, 2, 0))
                plt.title(f'epoch: {epoch}')
                plt.savefig(images_path / f'training_progress_{epoch}', format='png')

    model.update_loss(epoch, len(val_loader))
    model.update_learning_rate()  # update learning rates in the beginning of every epoch.

    # calculate FID. TODO: remove after asserting the correlation with loss components
    fid_score = get_fid_score(config, model, batch_size=config.data_loader.batch_size)

    gpu_list = []
    for gpu in gpus:
        gpu_list.append((
            gpu.id,
            gpu.name,
            f"{gpu.load * 100:.1f}%",
            f"{gpu.memoryTotal}MB",
            f"{gpu.memoryUsed}MB",
            f"{gpu.memoryFree}MB",
            f"{gpu.temperature} °C"
        ))

    print(tabulate(gpu_list, headers=("ID", "Name", "Load", "Total Memory", "Used Memory", "Free Memory", "Temperature")))
    # with torch.profiler.profile(
    #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #         record_shapes=True
    # ) as prof:
    #     print(prof.key_averages().table(sort_by="cuda_time_total"))

    if fid_score < best_fid_score:
        model.save_networks("best", path_to_save / "checkpoints")
        best_fid_score = fid_score
        print(f'Best FID score: {best_fid_score}')

model.save_networks("last", path_to_save / "checkpoints")