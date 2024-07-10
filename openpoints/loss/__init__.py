from .cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from .lovasz_losses import lovasz_softmax
from .contrast_loss import SupConLoss, SelfInfoNCE, PointInfoNCE
from .build import build_criterion_from_cfg