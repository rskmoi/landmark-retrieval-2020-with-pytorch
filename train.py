import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from dataset.landmark_dataset import LandmarkDataset
from model.model import arcface_model
from metric.loss import ArcFaceLoss
from tqdm import tqdm
from pathlib import Path
import hydra
from omegaconf import DictConfig
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_one_epoch(loader: DataLoader,
                    model: torch.nn.Module,
                    optimizer: Optimizer,
                    criterion: torch.nn.Module,
                    epoch: int,
                    out: str):
    """
    Train 1 epoch.
    """
    model.train()
    pbar = tqdm(loader, total=len(loader))
    for step, sample in enumerate(pbar):
        images, labels = sample['image'].type(torch.FloatTensor).to(DEVICE), \
                         sample['label'].type(torch.LongTensor).to(DEVICE)
        optimizer.zero_grad()
        cosine = model(images)
        loss = criterion(cosine, labels)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.data.cpu().numpy(), epoch=epoch)
        if (step + 1) % 5000 == 0:
            torch.save(model.state_dict(), Path(out) / f"{epoch}epoch_{step}_step.pth")

    torch.save(model.state_dict(), Path(out) / f"{epoch}epoch_final_step.pth")


@hydra.main(config_path="config/config.yaml")
def train(cfg: DictConfig):
    """
    Entry point of training.
    :param cfg: Config of training, parsed by hydra.
    :return: None
    """
    out_dir = Path(cfg.path.output)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    dataset = LandmarkDataset(batch_size=cfg.train.batch_size, mode="train")
    model:torch.nn.Module = arcface_model(num_classes=dataset.dataset.num_classes,
                                          backbone_model_name=cfg.model.name,
                                          head_name=cfg.model.head,
                                          extract_feature=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = ArcFaceLoss()

    for epoch in range(cfg.train.epochs):
        train_one_epoch(loader=dataset.get_loader(),
                        model=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        epoch=epoch,
                        out=out_dir)


if __name__ == '__main__':
    train()