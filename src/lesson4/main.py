import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import ModelNetLightningDataset
from model import PointNetLightning


@hydra.main(config_path=".", config_name="conf", version_base=None)
def main(cfg: DictConfig):
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)

    dataset = ModelNetLightningDataset(
        root=cfg.data.root,
        num_points=cfg.data.num_points,
        num_classes=cfg.data.num_classes,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )

    model = PointNetLightning(cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename="pointnet-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=2,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        patience=cfg.training.early_stopping_patience,
        mode="max",
        verbose=True,
    )

    logger = TensorBoardLogger(
        save_dir=cfg.training.log_dir, name="pointnet", version=None
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.device,
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=cfg.training.gradient_clip_val,
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(model, datamodule=dataset)

    trainer.test(model, ckpt_path="best")

    print("\nTraining completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"TensorBoard logs at: {logger.log_dir}")


if __name__ == "__main__":
    main()
