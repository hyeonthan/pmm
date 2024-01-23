"""Import libraries"""
import os
from datetime import datetime
import pytz

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.metrics import *

from lightning.pytorch import seed_everything, Trainer

from lightning.pytorch.loggers import TensorBoardLogger

from dataloaders.bcic4a import BCICompet2aIV

from models.litmodel import LitModel
from models.init import get_model

from utils.training_utils import get_configs

""" Config setting"""
CONFIG_PATH = f"{os.getcwd()}/configs"
filename = "config.yaml"

with open(f"{CONFIG_PATH}/{filename}") as file:
    args = get_configs(config_path=CONFIG_PATH, filename=filename)
    KCT = pytz.timezone("Asia/Seoul")
    args.current_time = datetime.now(KCT).strftime("%Y%m%d-%H:%M:%S")

cudnn.benchmark = True
cudnn.fastest = True
cudnn.deteministic = True


args.lr = float(args.lr)
args.weight_decay = float(args.weight_decay)

seed_everything(args.SEED)


def load_data():
    args.domain_type = "source"
    source_data = BCICompet2aIV(args)
    args.domain_type = "target"
    target_data = BCICompet2aIV(args)

    args.domain_type = "test"
    test_data = BCICompet2aIV(args)

    source_dataloaders = DataLoader(source_data, shuffle=True, batch_size=args.batch)
    target_dataloaders = DataLoader(target_data, shuffle=True, batch_size=args.batch)
    test_dataloaders = DataLoader(test_data, shuffle=True, batch_size=args.batch)

    return [source_dataloaders, target_dataloaders], test_dataloaders


def load_model():
    encoder = get_model(args=args)
    model = LitModel(model=encoder, args=args)
    return model


def main():
    ### Load Model ###
    model = load_model()

    ### Load Data ###
    train_dataloaders, test_dataloaders = load_data()

    ### Load logger and callbacks ###
    logger = TensorBoardLogger(
        args.LOG_PATH,
        name=f"{args.experiment_name}/{args.current_time}_{args.model_name}_cls",
    )
    devices = list(map(int, args.GPU_NUM.split(",")))

    trainer = Trainer(
        max_epochs=args.EPOCHS,
        strategy="ddp_find_unused_parameters_true",
        devices=devices,
        accelerator="gpu",
        logger=logger,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloaders,
        val_dataloaders=test_dataloaders,
    )


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception as e:
        print(e)
        print(traceback.format_exc())
