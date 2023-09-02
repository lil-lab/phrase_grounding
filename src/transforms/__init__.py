from .pixelbert import (
    pixelbert_transform,
    pixelbert_transform_noresize,
    pixelbert_transform_randaug,
    pixelbert_square
)

from .mdetr import (
    mdetr_transform_minmax,
    mdetr_touchdown_noresize,
    mdetr_transform_square,
    mdetr_transform_randaug,
)


_transforms = {
    "pixelbert": pixelbert_transform,
    "pixelbert_noresize": pixelbert_transform_noresize,
    "pixelbert_randaug": pixelbert_transform_randaug,
    "pixelbert_square": pixelbert_square,
    "mdetr": mdetr_transform_minmax,
    "mdetr_touchdown_val": mdetr_touchdown_noresize,
    "mdetr_randaug": mdetr_transform_randaug,
    "mdetr_square": mdetr_transform_square,
}


def keys_to_transforms(keys: list, size=224):
    return [_transforms[key](size=size) for key in keys]
