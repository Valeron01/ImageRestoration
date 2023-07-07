import cv2
import timm.data
import torch
import torchvision.models
from torch import nn
from timm.models.vgg import vgg19


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        model = vgg19(pretrained=True, features_only=True)
        self.vgg = nn.Sequential(*list(model.values())[:-1])
        self.vgg = self.vgg.requires_grad_(False).eval()
        self.register_buffer("mean", torch.FloatTensor(timm.data.IMAGENET_DEFAULT_MEAN)[None, :, None, None])
        self.register_buffer("std", torch.FloatTensor(timm.data.IMAGENET_DEFAULT_STD)[None, :, None, None])

    def preprocess(self, x):
        x = torch.flip(x, [1])
        return (x - self.mean) / self.std

    def forward(self, prediction, target):
        prediction = self.preprocess(prediction)
        target = self.preprocess(target)
        return nn.functional.mse_loss(self.vgg(prediction) / 12.75, self.vgg(target) / 12.75)


if __name__ == '__main__':
    loss = VGGLoss()
    model = timm.models.vgg.vgg19(pretrained=True)
    image = cv2.imread(r"H:\ExtractedDatasets\Coco\val2017\000000000285.jpg")
    image = torch.from_numpy(image).float().permute(2, 0, 1)[None] / 255.0
    image = loss.preprocess(image)

    prediction = model(image)
    print(torchvision.models.vgg._IMAGENET_CATEGORIES[prediction.argmax(dim=1).item()])
