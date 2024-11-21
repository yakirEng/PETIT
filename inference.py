from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
import torch
from datetime import datetime
from PIL import Image

from src.cycle_gan_model import CycleGANModel
from src.cut_model import CUTModel
from src.utils.deep import NetPhase
from src.configs import TestConfig
from src.dataset import MonoDS


def main():
    config = TestConfig()
    model = initialize_petit_model(config)


    dataset = MonoDS(src_dir=Path("data", "pan", "test"))

    dataloader = create_dataloader(config, dataset)

    fmt_of_output = "png"
    path_to_save = Path.cwd() / 'results' / 'transformed' / datetime.now().strftime("%Y%m%d_h%Hm%Ms%S")
    path_to_save.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for i, data in enumerate(tqdm(dataloader, desc="Test")):
            model.set_input(data)
            model.forward()

            visuals = model.get_current_visuals()

            for type in ["pan_real", "mono_fake", "mono_phys"]:
                sub_dir = path_to_save / type
                sub_dir.mkdir(parents=True, exist_ok=True)
                cur_vis = visuals[type]
                domain = type.split("_")[0]
                images = model.rec_image(cur_vis, domain=domain, fmt=fmt_of_output)
                if fmt_of_output == "npy":
                    for j, image in enumerate(images):
                        np.save(f"{str(sub_dir)}/{i * batch_size + j}.npy", image)
                else:
                    for j, image in enumerate(images):
                        pil_img = Image.fromarray(image)
                        pil_img.save(f"{str(sub_dir)}/{i * batch_size + j}.{fmt_of_output}")



def create_dataloader(config, dataset):
    batch_size = 100
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config.data_loader.num_threads),
    )
    print(f"The number of images for inference = {len(dataloader) * batch_size}")
    return dataloader


def initialize_petit_model(config):
    # config.model = "CUT" # "CycleGan"
    # config.thermal.is_fpa_input = True
    # config.thermal.is_physical_model = True

    backbone = CUTModel if config.model == "CUT" else CycleGANModel
    model = backbone(config)

    model.set_phase(NetPhase.test)
    model.setup(config)
    model.load_networks("best")

    return model


