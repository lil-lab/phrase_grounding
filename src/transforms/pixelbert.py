# source: https://github.com/dandelin/ViLT
from .utils import (
    inception_normalize,
    MinMaxResize,
    NoResizeRound,
    SquareResize
)
from torchvision import transforms
from .randaug import RandAugment
import src.transforms.mdetr_transfrom as T

def pixelbert_square(size=512):
    return transforms.Compose(
        [
            SquareResize(size=size), # this is equivalent to min_max resize 
            transforms.ToTensor(),
            inception_normalize,
        ]
    )

def pixelbert_transform(size=800):
    longer = int((1333 / 800) * size)

    return transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )

def pixelbert_transform_noresize(size=-1):
    return transforms.Compose(
        [
            NoResizeRound(),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )

def pixelbert_transform_randaug(size=800):
    longer = int((1333 / 800) * size)
    trs = transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))
    return trs


