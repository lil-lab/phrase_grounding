from .coco_format_datamodule import CocoFormatDataModule

_datamodules = {
    "touchdown_sdr": CocoFormatDataModule,
    "f30k_refgame": CocoFormatDataModule,
    "tangram_refgame": CocoFormatDataModule,
}
