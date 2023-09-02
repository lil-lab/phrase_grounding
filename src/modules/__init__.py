from .vilt_module import ViLTransformerSS, ViLTransformerProbeSS
from .mdetr_module import MDETR

model_cls = {
    "vilt_aligner": ViLTransformerSS,
    "vilt_aligner_probe": ViLTransformerProbeSS,
    "mdetr": MDETR,
}
