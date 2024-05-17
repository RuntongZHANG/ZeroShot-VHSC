from models.segmentor.attrPromptclip import AttrPromptCLIP
from models.segmentor.zegclip import ZegCLIP

from models.backbone.text_encoder import CLIPTextEncoder
from models.backbone.img_encoder import VPTCLIPVisionTransformer, VPTCLIPAttrPromptVisionTransformer
from models.decode_heads.decode_seg import ATMSingleHeadSeg
from models.decode_heads.decode_seg_attr import ATMSingleHeadSegAttr

from models.losses.atm_loss import SegLossPlus, SegLossOurs

from configs._base_.datasets.dataloader.vsac_attr20_bg import VSACDatasetAttr20BG

