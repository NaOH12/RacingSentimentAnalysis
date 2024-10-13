from lightning import Trainer
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from datasets.acc_dataset import ACCDataset
from engine.vae_lit_module import LitVAEModule

if __name__ == '__main__':
    num_frames = 100

    # Define dataloaders
    dataset = ACCDataset(num_frames=num_frames)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, persistent_workers=True)
    # train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, persistent_workers=True)

    # Define the model
    model = LitVAEModule(num_frames=num_frames, latent_dim=2)

    wandb_logger = WandbLogger(log_model="all", project="RacingSentimentAnalysis", )

    # Define the trainer
    trainer = Trainer(
        max_epochs=1, logger=wandb_logger,
        callbacks=[
            ModelSummary(max_depth=4),
            ModelCheckpoint(dirpath='checkpoints/')
        ],
        # limit_val_batches=0
        limit_train_batches=1000,
    )
    trainer.fit(model=model, train_dataloaders=train_loader)
