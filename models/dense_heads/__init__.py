from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .det_head import DetHead
from .mmdet_2d_head import MMDet2DHead

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'DetHead': DetHead,
    'MMDet2DHead': MMDet2DHead,
}
