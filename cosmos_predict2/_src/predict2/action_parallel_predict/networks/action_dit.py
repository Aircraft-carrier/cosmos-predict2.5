import torch
import torch.nn as nn
import torch.amp as amp
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple, Dict, Any
from einops import rearrange
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import MiniTrainDIT, Block, VideoSize, Attention, SACConfig
import transformer_engine as te
import math
"""
Run this command to interactively debug:
PYTHONPATH=. python cosmos_predict2/_src/predict2/action_parallel_predict/networks/action_dit.py
TODO: (zsh 2025-12-16)
- [x] Step 1: Successfully initialize a MinimalV1LVGDiT model instance using Hydra
- [x] Step 2: Implement a dummy action data loader that generates synthetic batches(shape,type,device)
- [-] Step 3: Debug MINItrainDIT and Extract Intermediate Layer Activations
- [x] Step 4: Build an ActionDiT module which could co-generate action and video
- [x] Step 5: Use different timestep sampling strategies for action and video generation. 
"""


class MaskedAttention(Attention):
    """
    A subclass of Attention that adds support for explicit attention masks.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        video_size: Optional["VideoSize"] = None,
    ):
        if attn_mask is None:
            return super().compute_attention(q, k, v, video_size=video_size)
        
        
        # Rearrange to [B, H, S_q, D] format expected by torch SDPA
        original_v_dtype = v.dtype
        
        q = q.transpose(1, 2)      # [B, H, S_q, D]
        k = k.transpose(1, 2)      # [B, H, S_k, D]
        v = v.transpose(1, 2).to(q.dtype)              # [B, H, S_k, D]
        
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            # True = keep, False = mask out → convert to -inf for masked positions
            attn_mask = attn_mask.to(q.device)
            attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill(~attn_mask, float('-inf'))

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,  
            is_causal=False
        ).to(original_v_dtype)  # [B, H, S_q, D]
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.output_dropout(self.output_proj(out))  
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,  
        video_size: Optional["VideoSize"] = None,
    ):
        """
        Args:
            x (Tensor): Query tensor of shape [B, Mq, K]
            context (Optional[Tensor]): Context (key/value) tensor of shape [B, Mk, K]
            rope_emb (Optional[Tensor]): RoPE embedding
            attn_mask (Optional[Tensor]): Attention mask.
                - Shape: [B, Mq, Mk] or [Mq, Mk]
                - Boolean: True = attend, False = ignore
                - Float: Additive mask (e.g., -inf for masked positions)
            video_size (Optional[VideoSize]): For spatial-temporal attention
        """
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        return self.compute_attention(q, k, v, attn_mask=attn_mask, video_size=video_size)      
    
class ActionRopePositionEmb(nn.Module):
    """
    Rotary Positional Embedding for action tokens (1D sequence).
    TODO：存在缺陷，无法与 图像文本 token 计算相对距离
    """
    def __init__(
        self,
        *,
        head_dim: int,
        max_action_len: int = 256,  
        s_extrapolation_ratio: float = 1.0,
        base: float = 10000.0,
        **kwargs,
    ):
        del kwargs  # for compatibility
        super().__init__()
        
        self.head_dim = head_dim
        self.max_action_len = max_action_len
        self.base = base
        self.s_ntk_factor = s_extrapolation_ratio ** (head_dim / (head_dim - 2)) if head_dim > 2 else 1.0

        # Precompute sequence indices [0, 1, ..., max_action_len - 1]
        self.register_buffer(
            "seq", 
            torch.arange(max_action_len, dtype=torch.float32),
            persistent=True
        )
        # Compute dim_range for frequency scaling: [0, 2, 4, ..., head_dim-2] / head_dim
        dim_range = torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim
        self.register_buffer("dim_range", dim_range, persistent=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Ensure buffers are on correct device (useful after load_state_dict)
        device = self.dim_range.device
        self.seq = torch.arange(self.max_action_len, dtype=torch.float32, device=device)
        dim_range = torch.arange(0, self.head_dim, 2)[: (self.head_dim // 2)].float().to(device) / self.head_dim
        self.dim_range = dim_range

    def generate_embeddings(
        self,
        num_action_tokens: int,
        s_ntk_factor: Optional[float] = None,
    ) -> torch.Tensor:
        assert num_action_tokens <= self.max_action_len, (
            f"num_action_tokens ({num_action_tokens}) exceeds max_action_len ({self.max_action_len})"
        )
        s_ntk_factor = s_ntk_factor if s_ntk_factor is not None else self.s_ntk_factor
        theta = self.base * s_ntk_factor
        freqs = 1.0 / (theta ** self.dim_range[:self.head_dim // 2])   # Compute frequencies: [d/2]
        half_emb = torch.outer(self.seq[:num_action_tokens], freqs)    # Outer product: [S] x [d/2] -> [S, d/2]
        emb = torch.cat([half_emb, half_emb], dim=-1)                  # [S, head_dim]
        return rearrange(emb, "s d -> s 1 1 d").float()     # Reshape to match video RoPE format: [S, 1, 1, head_dim]

    @property
    def seq_dim(self) -> int:
        return 0   

class ActionFinalLayer(nn.Module):
    """
    The final layer for action generation DiT.
    Input:  [B, T, D]  (D = hidden_size)
    Output: [B, T, action_dim]
    """

    def __init__(
        self,
        hidden_size: int,
        action_dim: int,                # e.g., 21*6=126 for SMPL-X hand pose
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        use_wan_fp32_strategy: bool = False,
    ):
        super().__init__()
        self.use_wan_fp32_strategy = use_wan_fp32_strategy
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.n_adaln_chunks = 2  # shift + scale
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim

        # Layer norm (no affine, modulated by AdaLN)
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Final linear projection to action space
        self.linear = nn.Linear(hidden_size, action_dim, bias=False)

        # AdaLN modulation network
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False),
            )

        self.init_weights()

    def init_weights(self) -> None:
        # Initialize final linear layer
        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.linear.weight, std=std, a=-3 * std, b=3 * std)

        # Initialize AdaLN modulation
        if self.use_adaln_lora:
            # First linear in modulation
            std_mod = 1.0 / math.sqrt(self.hidden_size)
            torch.nn.init.trunc_normal_(
                self.adaln_modulation[1].weight, std=std_mod, a=-3 * std_mod, b=3 * std_mod
            )
            # Second linear: zero init (common for residual-style modulation)
            torch.nn.init.zeros_(self.adaln_modulation[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation[1].weight)

        # Reset LayerNorm (though it has no learnable params due to affine=False)
        self.layer_norm.reset_parameters()

    def forward(
        self,
        x_B_T_D: torch.Tensor,                     # [B, T, D]
        emb_B_T_D: torch.Tensor,                   # timestep/action-type embedding [B, T, D]
        adaln_lora_B_T_2D: Optional[torch.Tensor] = None,  # [B, T, 2*D] if used
    ) -> torch.Tensor:
        """
        Returns:
            action_pred: [B, T, action_dim]
        """
        if self.use_wan_fp32_strategy:
            assert emb_B_T_D.dtype == torch.float32, "emb must be float32 under WAN FP32 strategy"

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            # Compute shift and scale from embedding
            if self.use_adaln_lora:
                assert adaln_lora_B_T_2D is not None, "adaln_lora must be provided when use_adaln_lora=True"
                mod_B_T_2D = self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_2D[:, :, : 2 * self.hidden_size]
            else:
                mod_B_T_2D = self.adaln_modulation(emb_B_T_D)

            shift_B_T_D, scale_B_T_D = mod_B_T_2D.chunk(2, dim=-1)  # each [B, T, D]

            # Apply AdaLN: y = LN(x) * (1 + scale) + shift
            x_norm_B_T_D = self.layer_norm(x_B_T_D)
            x_mod_B_T_D = x_norm_B_T_D * (1 + scale_B_T_D) + shift_B_T_D

            # Final projection to action space
            action_pred_B_T_A = self.linear(x_mod_B_T_D)  # [B, T, action_dim]

        return action_pred_B_T_A
    
class ActionAwareBlock(Block):
    """
    Extends Block to support action tokens with unidirectional attention:
    - Image tokens CANNOT attend to action tokens.
    - Action tokens CAN attend to all tokens (image + action).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_attn = MaskedAttention(
            kwargs.get("x_dim"),
            None,
            kwargs.get("num_heads"),
            kwargs.get("x_dim") // kwargs.get("num_heads"),
            qkv_format="bshd",
            backend=kwargs.get("backend","transformer_engine"),
            use_wan_fp32_strategy=kwargs.get("use_wan_fp32_strategy",False),
        )
        self.adaln_lora_dim = kwargs.get("adaln_lora_dim", 256)
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(2 * self.x_dim, self.adaln_lora_dim, bias=False),     # 1226 modified 替换为 2 x_dim
                nn.Linear(self.adaln_lora_dim, 3 * self.x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(2 * self.x_dim, self.adaln_lora_dim, bias=False),
                nn.Linear(self.adaln_lora_dim, 3 * self.x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(2 * self.x_dim, self.adaln_lora_dim, bias=False),
                nn.Linear(self.adaln_lora_dim, 3 * self.x_dim, bias=False),
            )

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,        # [B, 5, 16, 20, 2048]
        emb_B_T_D: torch.Tensor,          # [B, 1, 2048]
        crossattn_emb: torch.Tensor,      # [B, 512, 1024]
        action_B_T_D: torch.Tensor,       # [B, 16, 2048]
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,  # [1616, 1, 1, 128]
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:    
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb
        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if self.use_adaln_lora:
                shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                    self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                    self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)    
                shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                    self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)                                
            else:
                shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(
                    emb_B_T_D
                ).chunk(3, dim=-1)
                shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                    self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
                )
                shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)
                  
        # [2, 1, 2048] -> [2, 1, 1, 1, 2048]
        _, T, H, W, _ = x_B_T_H_W_D.shape
        L = T * H * W
        _, S, _ = action_B_T_D.shape
        # Helper: expand [B, 1, D] -> [B, 1, 1, 1, D] for video, or [B, S, D] for action
        dtype = x_B_T_H_W_D.dtype
        video_size = VideoSize(T=T, H=H, W=W)
        if self.cp_size is not None and self.cp_size > 1:
            video_size = VideoSize(T=T * self.cp_size, H=H, W=W)
            
        def _fn(unified_B_L_D, _norm_layer, _scale_B_T_D, _shift_B_T_D):
            return _norm_layer(unified_B_L_D) * (1 + _scale_B_T_D) + _shift_B_T_D         
        
        unified_B_L_D = torch.cat([
            rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d"),   # [B, 1600, 2048] 5 * 16 * 20
            action_B_T_D                                          # [B, 16, 2048]
        ], dim=1)                                                 # [B, 1616, 2048] 
        
        normalized_B_L_D = _fn(                                   # [B, 1616, 2048]
            unified_B_L_D,
            self.layer_norm_self_attn,                            # len(list(self.layer_norm_self_attn.parameters())) = 0 
            scale_self_attn_B_T_D.to(dtype),
            shift_self_attn_B_T_D.to(dtype),
        )    
        
        L_all = L + S  
        attn_mask = torch.ones(L_all, L_all, dtype=torch.bool)
        attn_mask[:L, L:] = False   
        
        result_B_L_D = self.self_attn(  # [B, 1616, 2048]
            normalized_B_L_D,           # q,k -> [2, 1616, 16, 128]
            context=None,
            rope_emb=rope_emb_L_1_1_D,  # [1616, 1, 1, 128]
            video_size=video_size, 
            attn_mask=attn_mask     
        )                              

        unified_B_L_D = unified_B_L_D + result_B_L_D * gate_self_attn_B_T_D.to(dtype)  # [B, 1616, 2048]

        def _x_fn(
            _x_L_D,
            layer_norm_cross_attn,
            _scale_cross_attn_B_T_D,
            _shift_cross_attn_B_T_D,
        ):
            _normalized_B_L_D = _fn(
                _x_L_D, layer_norm_cross_attn, _scale_cross_attn_B_T_D, _shift_cross_attn_B_T_D
            )
            _result_B_T_H_W_D = self.cross_attn(
                _normalized_B_L_D,
                crossattn_emb,
                rope_emb=rope_emb_L_1_1_D,
            )
            return _result_B_T_H_W_D

        
        result_B_L_D = _x_fn(   # [B, 1616, 2048]
            unified_B_L_D,
            self.layer_norm_cross_attn,
            scale_cross_attn_B_T_D.to(dtype),
            shift_cross_attn_B_T_D.to(dtype)
        )
        unified_B_L_D = unified_B_L_D + result_B_L_D * gate_cross_attn_B_T_D.to(dtype)  # [B, 1616, 2048]

        normalized_B_L_D = _fn(
            unified_B_L_D,
            self.layer_norm_mlp,
            scale_mlp_B_T_D.to(dtype),
            shift_mlp_B_T_D.to(dtype),
        )
        
        result_B_L_D = self.mlp(normalized_B_L_D)
        unified_B_L_D = unified_B_L_D + result_B_L_D * gate_mlp_B_T_D.to(dtype)

        N_vid = T * H * W
        x_B_N_D = unified_B_L_D[:, :N_vid, :]        # [B, 1600, 2048]
        x_B_T_H_W_D = rearrange(
            x_B_N_D,
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        action_B_T_D = unified_B_L_D[:, N_vid:, :]  # [B, 16, 2048]
                
        return x_B_T_H_W_D, action_B_T_D        
      

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.init_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.in_features)
        torch.nn.init.trunc_normal_(self.fc1.weight, std=std, a=-3 * std, b=3 * std)
        if self.fc1.bias is not None:
            torch.nn.init.zeros_(self.fc1.bias)
        std = 1.0 / math.sqrt(self.hidden_features)
        torch.nn.init.trunc_normal_(self.fc2.weight, std=std, a=-3 * std, b=3 * std)
        if self.fc2.bias is not None:
            torch.nn.init.zeros_(self.fc2.bias) 
               
    
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import Timesteps, TimestepEmbedding
    
class DualTimestepEncoder(nn.Module):
    def __init__(self, model_channels: int = 512, use_adaln_lora: bool = False):
        super().__init__()
        self.model_channels = model_channels
        self.sinusoidal_pos_emb = Timesteps(model_channels)
        self.proj = TimestepEmbedding(2 * model_channels, model_channels, use_adaln_lora=use_adaln_lora)
        # 1226 modified

    def forward(self, t1, t2):
        # Encode each timestep independently using sinusoidal positional embedding
        temb1 = self.sinusoidal_pos_emb(t1)
        temb2 = self.sinusoidal_pos_emb(t2)
        # Concatenate the two embeddings along the feature dimension 
        temb = torch.cat([temb1, temb2], dim=-1)  # [B, 2 * model_channels]  #
        return self.proj(temb)
    
    def init_weights(self) -> None:
        """Initialize MLP weights so that it initially acts as an identity mapping for the first half of inputs."""
        self.proj.init_weights()
                
from cosmos_predict2._src.predict2.action_parallel_predict.utils.utils import print_module_params

class ActionGenerationMinimalV1LVGDiT(MiniTrainDIT):  
    def __init__(self, *args, timestep_scale: float = 1.0, **kwargs):
        assert "in_channels" in kwargs, "in_channels must be provided"
        kwargs["in_channels"] += 1  # Add 1 for the condition mask
        action_dim = kwargs.get("action_dim", 14)
        if "action_dim" in kwargs:
            del kwargs["action_dim"]
            
        self.timestep_scale = timestep_scale
        log.critical(f"timestep_scale: {timestep_scale}")
        super().__init__(*args, **kwargs) 
        self.t_embedder = DualTimestepEncoder(
            model_channels=kwargs.get("model_channels", 2048),
            use_adaln_lora=kwargs.get("use_adaln_lora", False),
        ) 
        self.t_embedding_norm = te.pytorch.RMSNorm(2 * kwargs.get("model_channels", 768), eps=1e-6)
        self.blocks = nn.ModuleList(           
            [
                ActionAwareBlock(
                    x_dim=kwargs.get("model_channels", 768),
                    context_dim=kwargs.get("crossattn_emb_channels", 1024),
                    num_heads=kwargs.get("num_heads", 16),
                    mlp_ratio=kwargs.get("mlp_ratio", 4),
                    use_adaln_lora=kwargs.get("use_adaln_lora", False),
                    adaln_lora_dim=kwargs.get("adaln_lora_dim", 256),
                    backend=kwargs.get("atten_backend", 1.0),
                    image_context_dim=None if kwargs.get("extra_image_context_dim", None) is None else kwargs.get("model_channels", 768),
                    use_wan_fp32_strategy=kwargs.get("use_wan_fp32_strategy", False),
                )
                for _ in range(kwargs.get("num_blocks", 10))
            ]
        )
        
        self.action_pos_embedder = ActionRopePositionEmb(
            head_dim=self.model_channels // self.num_heads,
        )
        
        self.action_embedder = Mlp(
            in_features=action_dim,
            hidden_features=self.model_channels * 4,
            out_features=self.model_channels,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )      
        
        self.action_head = ActionFinalLayer(
            hidden_size=self.model_channels,
            action_dim=action_dim,
            use_adaln_lora=kwargs.get("use_adaln_lora", False),
            adaln_lora_dim=kwargs.get("adaln_lora_dim", 256),
            use_wan_fp32_strategy=kwargs.get("use_wan_fp32_strategy", False),
        )  
    
        # self.freeze_for_head_only_finetuning()  # 临时冻结方法，仅微调最后一层和动作头
        
    # LINK: cosmos_predict2.src.predict2.models.text2world_model_rectified_flow.py:205 
    def init_weights(self):
        self.x_embedder.init_weights()
        self.pos_embedder.reset_parameters()

        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.reset_parameters()

        if hasattr(self.t_embedder, 'proj'):
            self.t_embedder.init_weights()
            self.action_pos_embedder.reset_parameters()
            self.action_embedder.init_weights()
            self.action_head.init_weights()
        else:
            self.t_embedder[1].init_weights()
            
        for block in self.blocks:
            block.init_weights()

        self.final_layer.init_weights()
        self.t_embedding_norm.reset_parameters()

        if self.extra_image_context_dim is not None:
            self.img_context_proj[0].reset_parameters()
        
    # def freeze_for_head_only_finetuning(self):
    #     """Freeze all parameters except for the last block and action head."""
    #     for p in self.parameters():
    #         p.requires_grad_(False)
    #     for p in self.blocks[-1].parameters():
    #         p.requires_grad_(True)
    #     for p in self.action_head.parameters():
    #         p.requires_grad_(True)
        
    #     trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
    #     total = sum(p.numel() for p in self.parameters())
    #     log.critical(f"Head-only finetuning: {trainable:,} / {total:,} params trainable "
    #              f"({100 * trainable / total:.2f}%)")
          
    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,                      # [B, 16, 5, 32, 40]
        timesteps_B_T: torch.Tensor,                    # [B, 1]
        crossattn_emb: torch.Tensor,                    # [B, 512, 100352]
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None, # [B, 1, 5, 32, 40]
        fps: Optional[torch.Tensor] = None,             # [50., 50.]
        padding_mask: Optional[torch.Tensor] = None,    # [B, 1, 256, 320]
        data_type: Optional[DataType] = DataType.VIDEO, 
        intermediate_feature_ids: Optional[List[int]] = None,
        img_context_emb: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,          # [2, 16, 14] # One future frame corresponds to one action.
        action_timesteps_B_T: Optional[torch.Tensor] = None,  # [B, 1]
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        del kwargs
        # NOTE: If video conditioning is used, we need to additionally concatenate the ground-truth video mask.
        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:       # [B, 16, 5, 32, 40] -> [B, 17, 5, 32, 40]
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )
        # NOTE: we need to scale the timesteps, which is added for rectified flow model
        timesteps_B_T = timesteps_B_T * self.timestep_scale # (x 0.001) [[408.],[108.]] -> [[0.4082],[0.1079]] [B,1]
        action_timesteps_B_T = action_timesteps_B_T * self.timestep_scale if action_timesteps_B_T is not None else timesteps_B_T # [B, 1]
        assert action is not None, "action must be provided"
        action_B_T_D = self.action_embedder(action)  # [B, 17, 2048]
        # NOTE: Save the intermediate layer features.
        intermediate_feature_ids = None
        # NOTE: Obtain x in the shape (B, T, H, W, D) via patch embedding, and acquire rop_emb and pos_emb.
        assert isinstance(data_type, DataType), (
            f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        )
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )   # x_B_T_H_W_D -> [B, 5, 16, 20, 2048],  rope_emb_L_1_1_D -> [1600, 1, 1, 128]
        action_rope_emb_L_1_1_D = self.action_pos_embedder.generate_embeddings(action.shape[1]) # [17, 1, 1, 128] 
        rope_emb_L_1_1_D = torch.concat([rope_emb_L_1_1_D, action_rope_emb_L_1_1_D], dim=0)     # [1617, 1, 1, 128]
        
        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)  # [B, 512, 1024]  
        
        if img_context_emb is not None:
            assert self.extra_image_context_dim is not None, (
                "extra_image_context_dim must be set if img_context_emb is provided"
            )
            img_context_emb = self.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb
        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)     # [[0.8023]]  # [[0.6691]]
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T, action_timesteps_B_T) # [B, 1, 2048] [B, 1, 6144]      
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)      
            
        # t_embedding_B_T_D -> [B, 1, 2048], adaln_lora_B_T_3D -> [2, 1, 6144]

        # for logging purpose
        affline_scale_log_info = {}
        affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
                f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
            )        

        intermediate_features_outputs = []
        for i, block in enumerate(self.blocks):
            x_B_T_H_W_D, action_B_T_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                context_input,
                action_B_T_D=action_B_T_D,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
            if intermediate_feature_ids and i in intermediate_feature_ids:
                x_reshaped_for_disc = rearrange(x_B_T_H_W_D, "b tp hp wp d -> b (tp hp wp) d")
                intermediate_features_outputs.append(x_reshaped_for_disc)
                
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D[..., :self.model_channels], adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)  # [B, 16, 5, 32, 40]
        action = self.action_head(action_B_T_D, t_embedding_B_T_D[..., self.model_channels:], adaln_lora_B_T_2D=adaln_lora_B_T_3D)        # [B, 17, 14]
        if intermediate_feature_ids:
            if len(intermediate_features_outputs) != len(intermediate_feature_ids):
                log.warning(
                    f"Collected {len(intermediate_features_outputs)} intermediate features, "
                    f"but expected {len(intermediate_feature_ids)}. "
                    f"Requested IDs: {intermediate_feature_ids}"
                )
            return x_B_C_Tt_Hp_Wp, action, intermediate_features_outputs
        # return x_B_C_Tt_Hp_Wp : [B, 16, 5, 32, 40] , action : [B, 17, 14]
        return x_B_C_Tt_Hp_Wp, action
    
# ========== For Debug ============== 
class DummyDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 1000,
        C: int = 16,
        T: int = 5,
        H: int = 32,
        W: int = 40,
        crossattn_seq_len: int = 512,
        crossattn_feat_dim: int = 100352,
        T_action: int = 16,
        D_action: int = 14,
        fps_val: float = 50.0,
    ):
        self.num_samples = num_samples
        self.C = C
        self.T = T
        self.H = H
        self.W = W
        self.crossattn_seq_len = crossattn_seq_len
        self.crossattn_feat_dim = crossattn_feat_dim
        self.T_action = T_action
        self.D_action = D_action
        self.fps_val = fps_val

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        B = 1
        x_BCTHW = torch.randn(B, self.C, self.T, self.H, self.W, dtype=torch.bfloat16).squeeze(0)  
        timesteps_B_T = torch.randint(0, 1000, (B, 1), dtype=torch.float32).float().squeeze(0)   
        crossattn_emb = torch.randn(B, self.crossattn_seq_len, self.crossattn_feat_dim, dtype=torch.bfloat16).squeeze(0)  # [seq_len, feat_dim] # torch.bfloat16
        condition_video_input_mask = torch.randn(B, 1, self.T, self.H, self.W, dtype=torch.bfloat16).squeeze(0) 
        selected_timesteps = [0]
        condition_video_input_mask[ :, selected_timesteps, :, :] = 1.0
        fps = torch.full((B,), self.fps_val, dtype=torch.bfloat16).squeeze(0) 
        padding_mask = torch.zeros(B, 1, 256, 320, dtype=torch.bfloat16).squeeze(0)   
        actions_BTD = torch.randn(B, self.T_action, self.D_action, dtype=torch.bfloat16).squeeze(0)  

        
        return {
            "x_B_C_T_H_W": x_BCTHW,  
            "timesteps_B_T": timesteps_B_T,
            "crossattn_emb": crossattn_emb,
            "condition_video_input_mask_B_C_T_H_W": condition_video_input_mask,
            "fps": fps,
            "padding_mask": padding_mask,
            "data_type": DataType.VIDEO,
            "action": actions_BTD,
        }
        
net_config = {
	'adaln_lora_dim': 256, 
	'atten_backend': 'minimal_a2a', 
	'concat_padding_mask': True, 
	'crossattn_emb_channels': 1024, 
	'crossattn_proj_in_channels': 100352, 
	'extra_per_block_abs_pos_emb': False, 
	'in_channels': 16, 
	'max_frames': 128, 
	'max_img_h': 240, 
	'max_img_w': 240, 
	'model_channels': 2048, 
	'num_blocks': 1,     # Debug 28
	'num_heads': 16, 
	'out_channels': 16, 
	'patch_spatial': 2, 
	'patch_temporal': 1, 
	'pos_emb_cls': 'rope3d', 
	'pos_emb_interpolation': 'crop', 
	'pos_emb_learnable': True, 
	'rope_enable_fps_modulation': False, 
	'rope_h_extrapolation_ratio': 3.0, 
	'rope_t_extrapolation_ratio': 1.0, 
	'rope_w_extrapolation_ratio': 3.0,
	'sac_config':SACConfig(every_n_blocks=1, mode="predict2_2b_720_aggressive"), 
	'timestep_scale': 0.001, 
	'use_adaln_lora': True, 
	'use_crossattn_projection': True, 
	'use_wan_fp32_strategy': True,
}

def main():
    tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}
    dataset = DummyDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    video_net = ActionGenerationMinimalV1LVGDiT(**net_config)
    video_net.to(memory_format=torch.preserve_format, **tensor_kwargs)

    for batch in dataloader:
        batch = {
            k: v.to(memory_format=torch.preserve_format, **tensor_kwargs) 
            if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        batch["data_type"] = DataType.VIDEO
        result = video_net(**batch)
        import ipdb; ipdb.set_trace()
        print(result)
        break

""" Run this command to interactively debug:
PYTHONPATH=. python cosmos_predict2/_src/predict2/action_parallel_predict/networks/action_dit.py
"""
if __name__ == "__main__":
    main()