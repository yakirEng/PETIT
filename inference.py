import numpy as np
from tqdm import tqdm
import torch

from datetime import datetime
from PIL import Image

from src.configs import TestConfig
from src.cycle_gan_model import CycleGANModel
from src.cut_model import CUTModel
from src.utils.deep import NetPhase
from src.dataset import MonoDS

def load_petit_model(config, wl):
    backbone = CUTModel if config.model == "CUT" else CycleGANModel
    model = backbone(config)
    model.set_phase(NetPhase.test)
    model.setup(config)
    model.load_networks("best")
    return model
def predict(mode='single', data_path = 'data'):
    if mode == "batch":
        return predict_batch(data_path)
    elif mode == "single":
        return predict_single(model, pan_img, data_path)
def predict_single(model, pan_img):
    with torch.inference_mode():
        model.set_input(pan_img)
        model.forward()
        visuals = model.get_current_visuals()
        cur_vis = visuals["pan_real"]
        domain = "pan"
        fmt = "npy"
        mono = model.rec_image(cur_vis, domain=domain, fmt=fmt)
    return mono
def predict_batch(data_path):
    dataset = MonoDS(src_dir=data_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.data_loader.batch_size,
        shuffle=False,
        num_workers=int(self.config.data_loader.num_threads))


    with torch.inference_mode():
        for i, data in enumerate(tqdm(dataloader, desc="Test")):
            self.model.set_input(data)
            self.model.forward()

            visuals = self.model.get_current_visuals()

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



