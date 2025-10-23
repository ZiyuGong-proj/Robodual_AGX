
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import torch
import random
import torch.nn as nn
import numpy as np
import transformers
from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp, PatchEmbed
from prismatic.models.policy.transformer_utils import CrossAttentionBlock, PerceiverResampler, MAPBlock

torch.set_printoptions(threshold=np.inf)

# the xformers lib allows less memory, faster training and inference
try:
    import xformers
    import xformers.ops
except:
    XFORMERS_IS_AVAILBLE = False

# from timm.models.layers.helpers import to_2tuple
# from timm.models.layers.trace_utils import _assert

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Attention Layers from TIMM                                      #
#################################################################################

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_lora=False, RoPE=False, attention_mode='math'):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attention_mode = attention_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.RoPE = RoPE
        if RoPE:
            from rotary_embedding_torch import RotaryEmbedding
            self.rotary_emb = RotaryEmbedding(dim = head_dim // 2)

    def forward(self, x, mask = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        if self.RoPE:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)
        
        if self.attention_mode == 'xformers': # cause loss nan while using with amp
            x = xformers.ops.memory_efficient_attention(q, k, v).reshape(B, N, C)

        elif self.attention_mode == 'flash':
            # cause loss nan while using with amp
            # Optionally use the context manager to ensure one of the fused kerenels is run
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(B, N, C) # require pytorch 2.0

        elif self.attention_mode == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale

            if mask is not None:
                _MASKING_VALUE = -1e+30 if attn.dtype == torch.float32 else -1e+4
                attn.masked_fill(mask.to(attn.device) == 0, _MASKING_VALUE)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)


        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, use_fp16=False):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if use_fp16:
            t_freq = t_freq.to(dtype=torch.float16)
        t_emb = self.mlp(t_freq)
        return t_emb



#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiffusionTransformerBlock(nn.Module):
    """
    A DiT tansformer block with adaptive layer norm zero (adaLN-Zero) and cross-attention conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn_temporal = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.cross_attn = CrossAttentionBlock(  v_dim=hidden_size,
                                                l_dim=hidden_size,             
                                                embed_dim=hidden_size,
                                                num_heads=num_heads,)

    def forward(self, x, c, context_embed, n_batches=1, attn_mask = None):
        # add-LN conditioning
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # Spatial(action-wise) attention
        x = x + gate_msa.unsqueeze(1) * self.attn_temporal(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask)

        # Cross-attn conditioning
        x = self.cross_attn(x, context_embed)

        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# Core Specialist Policy Implementation
class DiT_SingleTokenAction(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        patch_size=1,
        in_channels=7,
        out_channels=7,
        hidden_size=1152,
        vis_dim=256,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_actions=8,
        num_obs_latents=8,
        cond_dropout_prob=0.1,
        attention_mode='math',
        in_context = False,
        with_proprio=True,
        with_depth=False,
        with_gripper=False,
        with_tactile=False,
        with_hist_action_num=4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels =  out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.in_context = in_context
        self.num_actions = num_actions
        self.cond_dropout_prob = cond_dropout_prob
        self.with_proprio = with_proprio
        self.with_depth = with_depth
        self.with_gripper = with_gripper
        self.with_tactile = with_tactile
        self.with_hist_action_num = with_hist_action_num

        self.x_embedder = nn.Linear(7 * 2, hidden_size, bias=True)

        if self.with_hist_action_num > 0:
            self.hist_act_embed = nn.Linear(7, hidden_size, bias=True)

        self.t_embedder = TimestepEmbedder(hidden_size)

        if with_proprio:
            self.proprio_embedder = nn.Sequential(
                nn.Linear(7, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )

        # 4096 -> hidden dim of the llama2 7b model
        self.context_adapter = nn.Linear(4096, hidden_size)

        self.visual_adapter = PerceiverResampler(
                                dim = hidden_size,
                                vis_dim = vis_dim,
                                depth = 1,
                                heads = num_heads,
                                num_latents = num_obs_latents,
                                num_media_embeds = 2,       # 1 history frame + 1 current frame
                            )

        if self.with_depth:
            self.depth_adapter = MAPBlock(n_latents = num_obs_latents, vis_dim = 256, embed_dim = hidden_size, n_heads = num_heads)

        if self.with_gripper:
            self.gripper_depth_adapter = MAPBlock(n_latents = num_obs_latents, vis_dim = 256, embed_dim = hidden_size, n_heads = num_heads)
            self.gripper_visual_adapter = PerceiverResampler(
                                            dim = hidden_size,
                                            vis_dim = vis_dim,
                                            depth = 1,
                                            heads = num_heads,
                                            num_latents = num_obs_latents,
                                            num_media_embeds = 1,
                                        )
        if self.with_tactile:
            self.tactile_adapter = MAPBlock(n_latents = 8, vis_dim = 256, embed_dim = hidden_size, n_heads = num_heads)
                                    
        # fixed sin-cos embedding (dummy value, replaced with RoPE ):
        self.temp_embed = nn.Parameter(torch.zeros(1, self.num_actions, hidden_size), requires_grad=False)
        self.hidden_size = hidden_size

        temporal_len = self.num_actions + self.with_hist_action_num 
        self.causal_mask = torch.tril(torch.ones(temporal_len, temporal_len)).view(1,1,temporal_len,temporal_len)


        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attention_mode=attention_mode, attn_drop=0.1, RoPE=True) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()



    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)


        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)



    # @torch.cuda.amp.autocast()
    # @torch.compile
    def forward(self, 
                x, 
                timesteps, 
                cond=None,
                visual_embedding=None, 
                depth_embedding=None,
                gripper_embedding=None,
                tactile_embedding=None,
                context=None,
                lang=None,
                proprio=None, 
                hist_action=None,
                cond_mask=None,
                use_fp16=False):
        """
        Forward pass of DiT.
        x: (N, T, d_action) tensor of actions
        t: (N,) tensor of diffusion timesteps
        cond: (N, T, d_cond) tensor of conditions
        """

        if len(timesteps.shape) == 0:
            timesteps = timesteps[None]
        timesteps = timesteps.expand(x.shape[0]).to(x.device)

        # cond[..., -1] = torch.where(cond[..., -1] > 0, 1., -1.)
        if use_fp16:
            x = x.to(dtype=torch.float16)

        x = torch.cat([cond, x], dim=-1) # ( b, T, d_action * 2 )
        batches, f, d_action = x.shape

        x = self.x_embedder(x.float()) 
        t = self.t_embedder(timesteps, use_fp16=use_fp16)      


        visual_embedding = self.visual_adapter(visual_embedding)
        visual_embedding = rearrange(visual_embedding, 'b n c d -> b (n c) d')


        if self.with_proprio:
            proprio = self.proprio_embedder(proprio)
            
    
        if depth_embedding is not None:
            visual_embedding = torch.cat([visual_embedding, self.depth_adapter(depth_embedding)], dim=1)
        

        if gripper_embedding is not None:
            gripper_visual_embedding = self.gripper_visual_adapter(gripper_embedding[0])
            gripper_visual_embedding = rearrange(gripper_visual_embedding, 'b n c d -> b (n c) d')
            gripper_depth_embedding  = self.gripper_depth_adapter(gripper_embedding[1])
            visual_embedding = torch.cat([visual_embedding, 
                                          gripper_visual_embedding, 
                                          gripper_depth_embedding,
                                          ], dim=1)
        

        if tactile_embedding is not None:
            tactile_embedding = self.tactile_adapter(tactile_embedding)
            visual_embedding = torch.cat([visual_embedding, tactile_embedding], dim=1)


        context = self.context_adapter(context)
        global_cond = context.mean(dim=1) if not self.with_proprio else proprio + context.mean(dim=1)


        if self.in_context:
            x = torch.cat([visual_embedding, x], dim=1)

        # Conditioned by history actions
        if self.with_hist_action_num > 0:
            x = torch.cat([self.hist_act_embed(hist_action), x], dim=1)

        # Classifier-free guidance
        if cond_mask is not None:
            global_cond = global_cond * cond_mask

        
        global_cond = global_cond + t 
        context_embed = context if self.in_context else torch.cat([visual_embedding, context], dim=1)
        for i in range(0, len(self.blocks)):
            x = self.blocks[i]( 
                            x = x, 
                            c = global_cond, 
                            n_batches = batches, 
                            context_embed = context_embed, 
                            attn_mask = self.causal_mask, 
                )


        if self.in_context or self.with_hist_action_num > 0:
            x = x[:, -f:]        # In Context Learning: Only return the pred noises

        x = self.final_layer(x, global_cond)                             

        return x





#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_temp_embed(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) 
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) 

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega 

    pos = pos.reshape(-1)  
    out = np.einsum('m,d->md', pos, omega) 

    emb_sin = np.sin(out) 
    emb_cos = np.cos(out) 

    emb = np.concatenate([emb_sin, emb_cos], axis=1) 
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################
def DiT_Large_STA(**kwargs):
    return DiT_SingleTokenAction(depth=18, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def DiT_Base_STA(**kwargs):
    return DiT_SingleTokenAction(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)

def DiT_Small_STA(**kwargs):
    return DiT_SingleTokenAction(depth=8, hidden_size=256, patch_size=1, num_heads=8, **kwargs)

def DiT_Tiny_STA(**kwargs):
    return DiT_SingleTokenAction(depth=6, hidden_size=256, patch_size=1, num_heads=8, **kwargs)

