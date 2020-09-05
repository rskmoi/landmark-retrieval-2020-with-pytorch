import click
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset.landmark_dataset import LandmarkDataset
from model.model import arcface_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate(embeds_path, labels_path):
    embeds = np.load(f"{embeds_path}.npy")
    labels = np.load(f"{labels_path}.npy")
    scores = []
    eval_pbar = tqdm(range(len(embeds)))
    for idx in eval_pbar:
        if idx % 11 != 0:
            continue
        q_vec = embeds[idx]
        cosine_distances = np.dot(q_vec, embeds.T)
        top11 = cosine_distances.argsort()[-11:][::-1]
        result_labels = labels[top11]
        score = 0.
        for i, result_label in enumerate(result_labels):
            if i == 0:
                continue
            if result_labels[0] == result_label:
                score += 0.1
        scores.append(score)
        eval_pbar.set_postfix(score=np.average(scores))


def save_features_and_labels(loader, model):
    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader, total=len(loader))
        feature_list = []
        label_list = []
        for step, sample in enumerate(pbar):
            images, labels = sample['image'].type(torch.FloatTensor).to(DEVICE), \
                             sample['label']
            label_list += labels
            cosine, feature = model(images)
            feature = F.normalize(feature, p=2, dim=1).cpu().numpy()
            feature_list.append(feature)

        features = np.concatenate(feature_list, axis=0)
        np.save("features", features)
        np.save("labels", np.array(label_list))


def _valid(pretrained_model_path: str):
    dataset = LandmarkDataset(batch_size=64, mode="test")
    model:torch.nn.Module = arcface_model(num_classes=dataset.dataset.num_classes,
                                          backbone_model_name="mobilenetv2",
                                          head_name="simple_head",
                                          extract_feature=True,
                                          pretrained_model_path=pretrained_model_path)
                                          # pretrained_model_path="outputs/2020-09-05/12-04-26/result/4th_exp/2epoch_final_step.pth")

    save_features_and_labels(loader=dataset.get_loader(),
                             model=model)


@click.command()
@click.option("--model_path", default="", help="path to model (.pth file).")
def valid(model_path):
    _valid(model_path)
    evaluate("features", "labels")


if __name__ == '__main__':
    valid()
