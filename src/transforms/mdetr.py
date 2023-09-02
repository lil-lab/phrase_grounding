
import src.transforms.mdetr_transfrom as T # TODO: figure out a way to do relative import here


normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def mdetr_transform_square(size=512):
    return T.Compose(
        [
            T.SquareResize(size=size), # this is equivalent to min_max resize 
            normalize
        ]
    )

def mdetr_transform_minmax(size=800):
    longer = int((1333 / 800) * size)
    scales = [size]

    return T.Compose(
        [
            T.RandomResize(scales, max_size=longer), # this is equivalent to min_max resize 
            normalize
        ]
    )

def mdetr_touchdown_noresize(size=800):
    longer = size
    scales = [800]

    return T.Compose(
        [
            T.RandomResize(scales, max_size=longer), # this is equivalent to min_max resize 
            normalize        
        ]
    )


def mdetr_transform_randaug(size=1333, cautious=True):
    longer = size
    if longer == 1856:
        # Touchdown low resolution
        scales = [256, 288, 320, 352, 384, 416]
    else:
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    return T.Compose(
        [   
            T.RandomSelect(
                T.RandomResize(scales, max_size=longer),
                T.Compose(
                    [
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, longer, respect_boxes=cautious),
                        T.RandomResize(scales, max_size=longer),
                    ]
                ),
            ),
            normalize
        ]
    )
