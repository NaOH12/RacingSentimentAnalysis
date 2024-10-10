from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from datasets.acc_dataset import ACCDataset
from engine.vae_lit_module import LitVAEModule

if __name__ == '__main__':
    # Define dataloaders
    dataset = ACCDataset()
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True)

    # Define the model
    model = LitVAEModule(num_frames=50, latent_dim=2)

    wandb_logger = WandbLogger(log_model="all", project="RacingSentimentAnalysis", )

    # Define the trainer
    trainer = Trainer(max_epochs=1, logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=train_loader)
