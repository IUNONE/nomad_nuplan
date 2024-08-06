import lightning as L
import yaml
import wandb
import time
from argparse import ArgumentParser
from nomad_nuplan_train.models.nomad.nomad import NoMaD
from nomad_nuplan_train.data.nuplan_dataset import Nuplan_Dataset
from torch.utils.data import DataLoader

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch import seed_everything

def main(config, args, wandb_logger):

    seed_everything(42, workers=True) # set seed

    #-------------------------- data ----------------------------------#

    train_set = Nuplan_Dataset(config['data_params'], config['model_params'], split='train')
    val_set = Nuplan_Dataset(config['data_params'], config['model_params'], split='val')
    
    train_loader = DataLoader(
        train_set,
        batch_size=config['train_params']["batch_size"],
        shuffle=True,
        num_workers=config['train_params']["num_workers"],
        drop_last=False,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config['train_params']["batch_size"],
        shuffle=False,
        num_workers=4,
        drop_last=True,
        persistent_workers=True,
    )
    
    print("\033[32m [ Sucess build Dataset! ] \033[0m" + f"num of train datapoints : {len(train_set)}, num of val datapoints : {len(val_set)}")

    #-------------------------- model --------------------------#

    nomad = NoMaD(
        config["data_params"],
        config["model_params"],
        config["train_params"],
    )
    # model_summary = ModelSummary(nomad, max_depth=-1)
    print("\033[32m [ Sucess build Model! ] \033[0m")
    # print(model_summary)

    #-------------------------- train & val --------------------------#

    trainer = L.Trainer( 
        accelerator = args.accelerator, 
        devices = args.devices,
        logger = wandb_logger,
        deterministic = True,
        max_epochs = config['train_params']["max_epochs"],
        gradient_clip_val = config['train_params']["grad_clip_max_norm"],
        # min_epochs=5,
        # default_root_dir=config['ckpt_path'],
        # strategy="deepspeed_stage_2",
        # precision=16
        # callbacks=[]
        # overfit_batches=1,
        # fast_dev_run=True, # runs 5 batch of training, validation, test and prediction for quick debug
    )

    print("\033[32m [ START TRAINING... ] \033[0m")
    trainer.fit(
        model=nomad, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        # ckpt_path = resume training
    )
    
    #-------------------------- test --------------------------#
    
    print("\033[32m [ START TEST... ] \033[0m")
    if config["test"]:
        test_set = Nuplan_Dataset(config['data_params'], config['model_params'], split='test')
        test_loader = DataLoader(
            test_set,
            batch_size=config['train_params']["batch_size"],
            shuffle=False,
            num_workers=4,
            drop_last=True,
            persistent_workers=True,
        )
        trainer.test(
            model=nomad, 
            dataloaders=test_loader,
        )

    print("\033[32m [ FINISH! ] \033[0m" + f"num batch: train - {trainer.num_training_batches}, val - {trainer.num_val_batches}")

if __name__ == "__main__":

    # -------------------------------cli args

    parser = ArgumentParser()
    parser.add_argument(
        "--accelerator", 
        default = 'gpu',
        help = "Accelerator (default: gpu)"
    )
    
    parser.add_argument(
        "--devices", 
        default = 1,
        type = int,
        help = "Number of devices (default: 1)"
    )
    
    parser.add_argument(
        "--batchsize", 
        default = 32,
        type = int,
        help = "batchsize (default: 32)"
    )

    args = parser.parse_args()

    # ---------------------------------load config file

    cfg_path = "config/nomad_nuplan_lightning.yaml"
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    
    config["train_params"]["batch_size"] = args.batchsize

    # ---------------------------------wandb logger
    
    config["wandb_params"]["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    wandb_params = config["wandb_params"]
    wandb_logger = WandbLogger(
        project=wandb_params["project_name"],
        log_model=True,     # Log model checkpoints at the end of training
        entity=wandb_params["entity"],
        settings=wandb.Settings(start_method="fork"),
        name=wandb_params["run_name"],
        )
    
    main(config, args, wandb_logger)