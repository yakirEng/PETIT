import numpy as np
from tqdm import tqdm
import torch

from PIL import Image

from src.dataset_code.mono.PETIT.src.cycle_gan_model import CycleGANModel
from src.dataset_code.mono.PETIT.src.cut_model import CUTModel
from src.dataset_code.mono.PETIT.src.utils.deep import NetPhase


def load_petit_model(test_config):
    backbone = CUTModel if test_config.model == "CUT" else CycleGANModel
    model = backbone(test_config)
    model.set_phase(NetPhase.test)
    model.setup(test_config)
    model.load_networks("best")
    return model

def get_petit_dataloader(test_config, dataset):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=test_config.data_loader.batch_size,
        shuffle=False,
        num_workers=int(test_config.data_loader.num_threads))
    return dataloader

def petit_predict(model, dataloader, fmt_of_output, path_to_save):
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



