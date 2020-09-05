import math

import torch
import torch.nn.functional as F
from torchvision.models import MobileNetV2, mobilenet_v2

from model.pooling import GeM

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_backbone(backbone_model_name: str):
    """
    Get backbone model and output feature size.
    TODO: Dynamically get the feature size.
    :param backbone_model_name:
    :return:
    """
    if backbone_model_name == "mobilenetv2":
        model: MobileNetV2 = mobilenet_v2(pretrained=True)
        feature_size = 1280
    else:
        raise ValueError
    return model.features, feature_size


def get_head(head_name: str, input_size: int, feature_size: int):
    if head_name == "simple_head":
        head = SimpleHead(input_size, feature_size)
    elif head_name == "original_head":
        head = OriginalHead(input_size, feature_size)
    else:
        raise ValueError
    return head


class SimpleHead(torch.nn.Module):
    """
    A simple head with a single fc layer.
    I (rskmoi) used this head in Landmark Retrieval 2020 and got 34th place.
    """
    def __init__(self, input_size: int, feature_size: int):
        super(SimpleHead, self).__init__()
        self.pooling = GeM()
        self.fc1 = torch.nn.Linear(input_size, feature_size)
        self.bn1 = torch.nn.BatchNorm2d(input_size)
        self.bn2 = torch.nn.BatchNorm1d(feature_size)

    def forward(self, x):
        x = self.bn1(x)
        x = F.dropout(x, p=0.5)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn2(x)
        return x


class OriginalHead(torch.nn.Module):
    """
    This Head is introduced in bestfitting's kaggle discussion.
    https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109
    """
    def __init__(self, input_size: int, feature_size: int):
        super(OriginalHead, self).__init__()
        self.pooling = GeM()
        self.bn1 = torch.nn.BatchNorm1d(input_size)
        self.fc1 = torch.nn.Linear(input_size, feature_size)
        self.bn2 = torch.nn.BatchNorm1d(feature_size)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Linear(feature_size, feature_size)
        self.bn3 = torch.nn.BatchNorm1d(feature_size)

    def forward(self, x):
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.bn1(x)
        x = F.dropout(x, p=0.25)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        x = self.bn3(x)
        return x


class ArcMarginProduct(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        """
        :param in_features: size of each input sample
        :param out_features: size of each output sample
        """
        super(ArcMarginProduct, self).__init__()
        self.weight = torch.nn.parameter.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight.cuda()))
        return cosine


class ArcFaceModel(torch.nn.Module):
    def __init__(self,
                 num_classes: int,
                 backbone_model_name: str,
                 head_name: str,
                 feature_size = 512,
                 extract_feature: bool = True):
        super(ArcFaceModel, self).__init__()
        self.extract_feature = extract_feature
        self.backbone, backbone_output_size = get_backbone(backbone_model_name)
        self.haed = get_head(head_name,
                             input_size=backbone_output_size,
                             feature_size=feature_size)
        self.arc_margin_product = ArcMarginProduct(feature_size, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        feature = self.haed(x)
        cosine = self.arc_margin_product(feature)
        if self.extract_feature:
            return cosine, feature
        else:
            return cosine


def arcface_model(num_classes: int,
                  backbone_model_name: str,
                  head_name: str,
                  extract_feature: bool,
                  pretrained_model_path: str = None) -> torch.nn.Module:
    model:torch.nn.Module = ArcFaceModel(num_classes=num_classes,
                                         backbone_model_name=backbone_model_name,
                                         head_name=head_name,
                                         extract_feature=extract_feature)

    model.to(DEVICE)

    if pretrained_model_path:
        model.load_state_dict(torch.load(pretrained_model_path))

    return model



if __name__ == '__main__':
    model = arcface_model(num_classes=2000,
                          backbone_model_name="mobilenetv2",
                          head_name="simple_head",
                          extract_feature=True)
    model.to(DEVICE)
    _in = torch.zeros(size=(4, 3, 224, 224)).cuda()
    cosine, feature = model(_in)
    print(cosine.size())
    print(feature.size())