import torch

from detectron2.config import configurable
from detectron2.modeling import ROI_BOX_HEAD_REGISTRY
from detectron2.layers import ShapeSpec

from ..backbone.twins import Twins


@ROI_BOX_HEAD_REGISTRY.register()
class TwinsBoxHead(Twins):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        **kwargs,
        ):
        super().__init__(**kwargs)
        self.patch_embeds = self.patch_embeds[-1:]
        self.blocks = self.blocks[-1:]
        self.pos_block = self.pos_block[-1:]
        self.pos_drops = self.pos_drops[-1:]
        
        if self.branch_embed:
            self.branch_embedding = self.branch_embedding[-1:]

        last_key = list(super().output_shape().keys())[-1]
        o = super().output_shape()[last_key]
        self._output_size = ShapeSpec(channels=o.channels, height=input_shape.height // 2, width=input_shape.width // 2)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['out_features'] = ['c5']
        ret['freeze_at'] = 0

        ret['input_shape'] = input_shape
        return ret
    
    @property
    @torch.jit.unused
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        return self._output_size

    def forward_single(self, x):
        B_x, _, _, _ = x.shape

        x, size_query = self.patch_embeds[-1](x)
        x = self.pos_drops[-1](x)

        for j, blk in enumerate(self.blocks[-1]):
            x = blk(x, size_query)
            
            if j == 0:
                x = self.pos_block[-1](x, size_query)  # PEG here
        
        x = self.norm(x)

        x = x.reshape(B_x, *size_query, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x, y=None):
        if y is None:
            return self.forward_single(x)
        B_x, _, _, _ = x.shape
        B_y, _, _, _ = y.shape

        x, size_query = self.patch_embeds[-1](x)
        x = self.pos_drops[-1](x)

        y, size_support = self.patch_embeds[-1](y)
        y = self.pos_drops[-1](y)

        for j, blk in enumerate(self.blocks[-1]):
            x, y = blk(x, size_query, y, size_support)
            
            if j == 0:
                x = self.pos_block[-1](x, size_query)  # PEG here
                y = self.pos_block[-1](y, size_support)
        
        x = self.norm(x)
        y = self.norm(y)

        x = x.reshape(B_x, *size_query, -1).permute(0, 3, 1, 2).contiguous()
        y = y.reshape(B_y, *size_support, -1).permute(0, 3, 1, 2).contiguous()
        return x, y
