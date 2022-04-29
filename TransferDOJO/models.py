import torchvision.models as models
from torch import nn
import torch
import torch.nn.functional as F


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_base(model, path, num_outputs):
    """
    load the base of the model
    :param model:
    :param path:
    :param num_outputs:
    :return:
    """
    model_state = torch.load(path)
    if isinstance(model_state, tuple):
        model_state = model_state[0]
    model.load_state_dict(model_state)
    # reset classifier to match size of problem at hand
    try:
        model.classifier = torch.nn.Linear(
            in_features=model.dn.classifier.in_features, out_features=num_outputs
        )
    except AttributeError:
        model.classifier = torch.nn.Linear(
            in_features=model.cin_features, out_features=num_outputs
        )
    return model


def freeze_base(model):
    """
    Freeze all weights except classsifier and resets weight of classifier
    """
    for para in model.parameters():
        para.requires_grad = False
    for para in model.classifier.parameters():
        para.requires_grad = True
    # randomize the linear classifer #
    for layer in model.classifier.modules():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()

    return model


class DenseNet169(nn.Module):
    def __init__(self, pretrained=True, num_outputs=2, one_channel=True):
        super(DenseNet169, self).__init__()
        self.dn = torch.hub.load(
            "pytorch/vision:v0.9.0", "densenet169", pretrained=pretrained
        )
        self.base = self.dn.features
        # take avg of weights for input layer #
        if one_channel:
            self.squeeze_model_to_one_channel()
        # self.batch_norm = torch.nn.BatchNorm2d(self.dn.classifier.in_features)
        self.classifier = torch.nn.Linear(
            in_features=self.dn.classifier.in_features, out_features=num_outputs
        )

    def forward(self, x):
        x = self.base(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # x = self.batch_norm(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def squeeze_model_to_one_channel(self):
        squeezed_conv0 = torch.nn.Conv2d(
            1,
            self.base[0].out_channels,
            kernel_size=self.base[0].kernel_size,
            stride=self.base[0].stride,
            padding=self.base[0].padding,
            bias=False,
        )

        squeezed_conv0.weight = torch.nn.Parameter(
            self.base[0].state_dict()["weight"].sum(axis=1, keepdim=True),
            requires_grad=True,
        )

        self.base[0] = squeezed_conv0


class ResNet18(nn.Module):
    def __init__(self, pretrained, num_outputs=2):
        super(ResNet18, self).__init__()
        self.base = models.resnet18(pretrained=pretrained)
        self.conv1 = self.base.conv1
        self.bn1 = self.base.bn1
        self.relu = self.base.relu
        self.maxpool = self.base.maxpool
        self.layer1 = self.base.layer1
        self.layer2 = self.base.layer2
        self.layer3 = self.base.layer3
        self.layer4 = self.base.layer4
        self.avgpool = self.base.avgpool
        self.__in_features = self.base.fc.in_features
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=num_outputs, bias=True),
        )

    def forward(self, x):
        x = self.feature_forward(x)
        x = self.classifier(x)
        return x

    def feature_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def feature_dim(self):
        return self.__in_features


