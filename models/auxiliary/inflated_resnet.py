from models.auxiliary.imagenet_pretraining import load_pretrained_2D_weights
from models.auxiliary.resnet.bottleneck import Bottleneck3D
from models.auxiliary.resnet.resnet import ResNet


def inflated_resnet(**kwargs):
    list_block = [Bottleneck3D, Bottleneck3D, Bottleneck3D, Bottleneck3D]
    list_layers = [3, 4, 6, 3]

    # Create the model
    model = ResNet(list_block,
                   list_layers,
                   **kwargs)

    # Pretrained from imagenet weights
    load_pretrained_2D_weights('resnet50', model, inflation='center')

    return model
