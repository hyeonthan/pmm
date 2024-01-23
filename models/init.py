import os
from typing import Optional, Callable
from glob import glob

from easydict import EasyDict

from models.backbones.deepconvnet import DeepConvNet
from models.backbones.djdan import DJDAN
from models.backbones.drda import GENERATOR, DISCRIMINATOR
from models.backbones.maan import MAAN

model_list = {
    "deepconvnet": DeepConvNet,
    "djdan": DJDAN,
    "drda": [GENERATOR, DISCRIMINATOR],
    "maan": MAAN,
}


def get_model(
    args: EasyDict,
    model_name: str = None,
    weight_path: str = None,
    load_ckpt: Optional[Callable] = None,
    **kwargs
):
    model = model_list[args.model_name]

    if isinstance(model, list):
        for i in range(len(model)):
            model[i] = model[i](args)
    else:
        model = model(args)

    return model
