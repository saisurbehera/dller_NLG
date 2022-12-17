import copy
import math
import os
import warnings
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

from functools import partial
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers import T5Config



from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoSelfAttention , GPTNeoAttention, GPTNeoMLP


"""
Got this from menzi configuration file
"""
from transformers.configuration_utils import PretrainedConfig
class RETROConfig(PretrainedConfig):
    model_type = "RETRO"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50257,
        max_position_embeddings=2048,
        hidden_size=2048,
        num_layers=24,
        attention_types=[[["global", "local"], 12]],
        num_heads=16,
        intermediate_size=None,
        window_size=256,
        activation_function="gelu_new",
        resid_dropout=0.0,
        embed_dropout=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        dec_cross_attn_layers=(5, 128),
        chunk_size=64,
        dim_head=64,
        enc_depth=2,
        enc_cross_attn_layers=None,
        enc_heads=16,
        dec_heads=16,
        enc_att_dropout=0.0,
        enc_ff_dropout=0.0,
        dec_attn_dropout=0.25,
        dec_ff_dropout=0.25,
        ff_mult=4,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.window_size = window_size
        self.activation_function = activation_function
        self.resid_dropout = resid_dropout
        self.embed_dropout = embed_dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.use_cache = use_cache

        self.enc_dim = hidden_size
        self.enc_depth = enc_depth
        self.chunk_size = chunk_size
        self.dim_head = dim_head
        self.dec_cross_attn_layers = dec_cross_attn_layers
        self.enc_cross_attn_layers = enc_cross_attn_layers
        self.enc_heads = enc_heads
        self.enc_att_dropout = enc_att_dropout
        self.enc_ff_dropout = enc_ff_dropout
        self.ff_mult = ff_mult
        self.dec_attn_dropout = dec_attn_dropout
        self.dec_heads = dec_heads
        self.dec_ff_dropout = dec_ff_dropout

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.attention_types = attention_types
        self.attention_layers = self.expand_attention_types_params(attention_types)


        if len(self.attention_layers) != self.num_layers:
                raise ValueError(
                    "Configuration for convolutional module is incorrect. "
                    "It is required that `len(config.attention_layers)` == `config.num_layers` "
                    f"but is `len(config.attention_layers) = {len(self.attention_layers)}`, "
                    f"`config.num_layers = {self.num_layers}`. "
                    "`config.attention_layers` is prepared using `config.attention_types`. "
                    "Please verify the value of `config.attention_types` argument."
                )

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
    

    def expand_attention_types_params(self,attention_types):
        attentions = []
        for item in attention_types:
            for _ in range(item[1]):
                attentions.extend(item[0])
        return attentions

# Built on top T5LayerNorm from the repo and modified for the RETRO-repo
class T5LayerNorm(nn.Module):
    def __init__(self, dims,  eps=1e-6):
        """
        Construct a layernorm module in the T5 style we modify by adding a scale norm

        This will become similar to RMSNorm from Apex
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.variance_epsilon = eps
        self.scale = dims ** -0.5

    def forward(self, hidden_states):

        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        ## We have to add a new component to scale it 

        # This is the original code, we modify it 
        # variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # # convert into half-precision if necessary
        # if self.weight.dtype in [torch.float16, torch.bfloat16]:
        #     hidden_states = hidden_states.to(self.weight.dtype)

        # return self.weight * hidden_states

        norm = hidden_states.norm(keepdim=True, dim=-1) * self.scale
        value = (hidden_states / norm.clamp(min=self.variance_epsilon)) * self.weight
        return value



"""
Pre Norm class
"""
class PreNorm(nn.Module):
    def __init__(self, dim, func):
        super().__init__()
        self.fn = func
        self.norm = T5LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs) + x

def exists(val):
    return val is not None


# Copied from https://github.com/lucidrains/RETRO-pytorch
def default(val, d):
    return val if exists(val) else d


# Copied from https://github.com/lucidrains/RETRO-pytorch
def apply_rotary_pos_emb(t, freqs):
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim=-1)


# Copied from https://github.com/lucidrains/RETRO-pytorch
def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


# Copied from https://github.com/lucidrains/RETRO-pytorch
def cast_tuple(val, num=1):
    return val if isinstance(val, tuple) else ((val,) * num)


# Copied from https://github.com/lucidrains/RETRO-pytorch
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, *, device, offset=0):
        seq = torch.arange(max_seq_len, device=device) + offset
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return rearrange(emb, 'n d -> 1 1 n d')



# Copied from https://github.com/lucidrains/RETRO-pytorch.retro_pytorch
class PostNorm(nn.Module):
    def __init__(self, dim, fn, scale_residual=1, norm_klass=T5LayerNorm):
        super().__init__()
        self.fn = fn
        self.scale_residual = scale_residual
        self.norm = norm_klass(dim)

    def forward(self, x, *args, **kwargs):
        residual = x * self.scale_residual
        out = self.fn(x, *args, **kwargs) + residual
        return self.norm(out)

# Copied from https://github.com/lucidrains/RETRO-pytorch
class ChunkedCrossAttention(nn.Module):
    def __init__(
        self,
        config,
        **kwargs
    ):
        super().__init__()
        dim = config.hidden_size
        dim_head = config.dim_head
        heads = config.dec_heads
        dropout = config.dec_attn_dropout
        self.chunk_size = config.chunk_size
        self.cross_attn = Attention(null_kv=True, dim=dim, dim_head=dim_head, heads=heads, dropout=dropout)

    def forward(self, x, *, context_mask=None, context, pos_emb=None):
        device = x.device
        # derive variables
        chunk_size = self.chunk_size

        b, n, num_chunks, num_retrieved = x.shape[0], x.shape[-2], *context.shape[-4:-2]

        # if sequence length less than chunk size, do an early return

        if n < self.chunk_size:
            return torch.zeros_like(x)

        # causal padding

        causal_padding = chunk_size - 1

        x = F.pad(x, (0, 0, -causal_padding, causal_padding), value=0.)

        # remove sequence which is ahead of the neighbors retrieved (during inference)

        seq_index = (n // chunk_size) * chunk_size
        x, x_remainder = x[:, :seq_index], x[:, seq_index:]

        seq_remain_len = x_remainder.shape[-2]

        # take care of rotary positional embedding
        # make sure queries positions are properly shifted to the future

        q_pos_emb, k_pos_emb = pos_emb
        q_pos_emb = F.pad(q_pos_emb, (0, 0, -causal_padding, causal_padding), value=0.)

        k_pos_emb = repeat(k_pos_emb, 'b h n d -> b h (r n) d', r=num_retrieved)
        pos_emb = (q_pos_emb.to(device), k_pos_emb.to(device))

        # reshape so we have chunk to chunk attention, without breaking causality

        x = rearrange(x, 'b (k n) d -> (b k) n d', k=num_chunks)
        context = rearrange(context, 'b k r n d -> (b k) (r n) d')

        if exists(context_mask):
            context_mask = rearrange(context_mask, 'b k r n -> (b k) (r n)')

        # cross attention
        if exists(context):
            context = context.to(device).to(dtype=x.dtype)
        if exists(context_mask):
            context_mask = context_mask.to(device).to(dtype=x.dtype)
        out = self.cross_attn(x, context=context, mask=context_mask, pos_emb=pos_emb)

        # reshape back to original sequence

        out = rearrange(out, '(b k) n d -> b (k n) d', b=b)

        # pad back to original, with 0s at the beginning (which will be added to the residual and be fine)

        out = F.pad(out, (0, 0, causal_padding, -causal_padding + seq_remain_len), value=0.)
        return out


# Since a lot of the skeleton code is based on HuggingFace T5 code, here is the load_tf_weights_folder
def load_tf_weights_in_retro_gpt(model, config, checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import re

        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        if "global_step" not in name and "adam" not in name:
            array = tf.train.load_variable(tf_path, name)
            array = tf.dtypes.cast(array.squeeze(), tf.float32).numpy()
            name = name.replace("attn/q", "attn/attention/q_proj/w")
            name = name.replace("attn/k", "attn/attention/k_proj/w")
            name = name.replace("attn/v", "attn/attention/v_proj/w")
            name = name.replace("attn/o", "attn/attention/out_proj/w")
            name = name.replace("norm_1", "ln_1")
            name = name.replace("norm_2", "ln_2")
            name = name.replace("attn/compute_output_bias/o_b", "attn/attention/out_proj/b")
            name = name.replace("conv1d_main/c_fc/kernel", "c_fc/w")
            name = name.replace("conv1d_main/c_fc/bias", "c_fc/b")
            name = name.replace("conv1d_main/c_proj/kernel", "c_proj/w")
            name = name.replace("conv1d_main/c_proj/bias", "c_proj/b")

            names.append(name)
            arrays.append(array)

    for name, array in zip(names, arrays):
        name = name[5:]  # skip "gpt2/"
        name = name.split("/")
        pointer = model.transformer
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]

        if name[-1] == "w" and name[-2] in ["out_proj", "k_proj", "q_proj", "v_proj", "c_proj", "c_fc"]:
            array = array.transpose()

        if name == ["wte"]:
            # if vocab is padded, then trim off the padding embeddings
            array = array[: config.vocab_size]

        if pointer.shape != array.shape:
            raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched {name}")

        print(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)

    # init the final linear layer using word embeddings
    embs = model.transformer.wte.weight
    lin = nn.Linear(embs.size()[1], embs.size()[0], bias=False)
    lin.weight = embs
    model.set_output_embeddings(lin)
    return model




# Copied from https://github.com/lucidrains/RETRO-pytorch
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        causal=False,
        dropout=0.,
        null_kv=False
    ):
        super().__init__()
        context_dim = default(context_dim, dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        # allowing for attending to nothing (null function)
        # and to save attention from breaking if all retrieved chunks are padded out
        self.null_k = nn.Parameter(torch.randn(inner_dim)) if null_kv else None
        self.null_v = nn.Parameter(torch.randn(inner_dim)) if null_kv else None

    def forward(self, x, mask=None, context=None, pos_emb=None):
        b, device, h, scale = x.shape[0], x.device, self.heads, self.scale

        kv_input = default(context, x)

        q, k, v = self.to_q(x), self.to_k(kv_input), self.to_v(kv_input)
        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # scale
        q = q * scale

        # apply relative positional encoding (rotary embeddings)
        if exists(pos_emb):
            q_pos_emb, k_pos_emb = cast_tuple(pos_emb, num=2)
            q = apply_rotary_pos_emb(q, q_pos_emb)
            k = apply_rotary_pos_emb(k, k_pos_emb)

        # add null key / values
        if exists(self.null_k):
            nk, nv = self.null_k, self.null_v
            nk, nv = map(lambda t: repeat(t, '(h d) -> b h 1 d', b=b, h=h), (nk, nv))
            k = torch.cat((nk, k), dim=-2)
            v = torch.cat((nv, v), dim=-2)

        # derive query key similarities
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # masking
        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            if exists(self.null_k):
                mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones(i, j, device=device, dtype=torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        # attention
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        # combine heads linear out
        return self.to_out(out)

"""
Although the Retro implementation of lucid brains uses it own trained encoder. 

We will be using a BERT encoder as it simplifies the results substantialy and allows us to use pretrained
blocks and we only add the final context representation at the final layer
"""

from transformers import BertModel

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(mult * dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        return self.ff(x)

# heavily modified from https://github.com/lucidrains/RETRO-pytorch
class RETROEncoderBERThead(nn.Module):
    def __init__(
        self,
        config,
        norm_class=T5LayerNorm,
        post_norm=False,
        scale_residual=1.0,
        causal=False,
        final_norm=True
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        enc_dim = config.enc_dim
        hidden_size = config.hidden_size
        enc_depth = config.enc_depth
        cross_attn_layers = config.enc_cross_attn_layers
        dim_head = config.dim_head
        context_dim = config.hidden_size
        enc_heads = config.enc_heads
        ff_mult = config.ff_mult
        attn_dropout = config.enc_att_dropout
        enc_ff_dropout = config.enc_ff_dropout
        # partial rotary embeddings, which is better than full rotary
        # Wang and Komatsuzaki et al https://github.com/kingoflolz/mesh-transformer-jax/
        rotary_emb_dim = min(dim_head, 32)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim)

        wrapper = partial(PreNorm, enc_dim) if not post_norm else partial(PostNorm, enc_dim, scale_residual=scale_residual)

        for layer_num in range(1, enc_depth + 1):
            has_cross_attn = not exists(cross_attn_layers) or layer_num in cross_attn_layers
            self.layers.append(nn.ModuleList([
                wrapper(Attention(dim=enc_dim, dim_head=dim_head, heads=enc_heads, dropout=attn_dropout, causal=causal)),
                wrapper(Attention(dim=enc_dim, context_dim=context_dim, dim_head=dim_head, heads=enc_heads, dropout=attn_dropout)) if has_cross_attn else None,
                wrapper(FeedForward(dim=enc_dim, mult=ff_mult, dropout=enc_ff_dropout)),
            ]))
        self.norm_out = T5LayerNorm(enc_dim) if final_norm and not post_norm else nn.Identity()
        self.project_out = nn.Linear(enc_dim, hidden_size) if exists(hidden_size) else nn.Identity()


    def forward(self, x, *, mask=None, chunked_seq):
        # x = self.BertModel(x)
        device, chunk_size, seq_len = x.device, x.shape[-2], chunked_seq.shape[-2]
        q_pos_emb = self.rotary_pos_emb(chunk_size, device=device)
        k_pos_emb = self.rotary_pos_emb(seq_len, device=device)
        for attn, cross_attn, ff in self.layers:
            x = attn(x, mask=mask, pos_emb=q_pos_emb)
            if exists(cross_attn):
                x = cross_attn(x, context=chunked_seq, pos_emb=(q_pos_emb, k_pos_emb))
            x = ff(x)
        x = self.norm_out(x)
        return self.project_out(x)





class RETROModelBlock(nn.Module):
    
    """
    In this class, we create a GPT style block containing attention blocks and a MLP layer. 
    to simplify the process, we use the GPTNeo blocks already called. 

    The class is based on how similar the GPTNeo Huggingface implementations. 

    So you will find it similar to this implementation at 
    https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L476

    the main difference is at every interval there is the decoder chunked attention blocks.

    Mengzi block is similar but we have a seperate positional embedding layer
    """

    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.inner_dim = config.intermediate_size if config.intermediate_size is not None else 4 * self.hidden_size

        # layers
        self.ln_1 = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPTNeoAttention(config = self.config , layer_id = self.layer_id)
        self.ln_2 = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPTNeoMLP(intermediate_size=self.inner_dim, config = self.config )

        self.hidden_size = config.hidden_size
        self.norm_class = T5LayerNorm
        self.dimension = config
        wrapper = partial(PostNorm, config.hidden_size, scale_residual=1, norm_klass=T5LayerNorm)
        if layer_id in config.dec_cross_attn_layers:
            self.cross_attn = wrapper(ChunkedCrossAttention(config))
        else:
            self.cross_attn = None
        self.chunk_size = config.chunk_size
        self.dim_head = config.dim_head

    """
    GPT-NEO conventionally uses hidden_states, attention_mask, head_mask, use_cache, output_attentions

    We add the retrieval,encoded resuts
    """
    def self(
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        retrieved_encoded = None,
        encoder = None,
        retrieval = None,
        encoder_retrieved_mask=None,
        context_mask=None,
        cross_attn_k_pos_emb = None,
    ):

        seq_len = hidden_states.shape[-2]
        num_seq_chunks = seq_len // self.chunk_size
        seq_index = num_seq_chunks * self.chunk_size
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attention_outputs = self.attention(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attention_outputs = attention_outputs[0]
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual
        
        hidden_states = self.ln_2(hidden_states)
        residual = hidden_states


        if exists(self.cross_attn) and exists(retrieval):
            num_chunks, num_neighbors, chunk_size = retrieval.shape[-4:-1]
            if not retrieved_encoded:
                retrieval = rearrange(retrieval, 'b k r n d -> (b k r) n d')
                seq_as_context = repeat(hidden_states[:, :seq_index], 'b (k n) d -> (b k r) n d', n=self.chunk_size, r=num_neighbors)
                retrieval = encoder(retrieval, mask=encoder_retrieved_mask, chunked_seq=seq_as_context)
                retrieval = rearrange(retrieval, '(b k r) n d -> b k r n d', k=num_chunks, r=num_neighbors)
                retrieved_encoded = True

            hidden_states = self.cross_attn(
                    hidden_states,
                    context=retrieval,
                    context_mask=context_mask,
                    pos_emb=cross_attn_pos_emb,
            )
        feed_forward_hidden_states = self.mlp(hidden_states)

        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs, retrieved_encoded



"""
Part of the code has been copied from GPT-NEO but it has been modified for fiting the 
case for language models
"""
class RETROPreTrainedModel(PreTrainedModel):
    config_class = RETROConfig
    load_tf_weights = load_tf_weights_in_retro_gpt
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RETROModelBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear,)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, RetroModel):
            module.gradient_checkpointing = value


class RetroModel(RETROPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.chunk_size = config.chunk_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(float(config.embed_dropout))
        self.h = nn.ModuleList([RETROModelBlock(config, layer_id=i) for i in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        rotary_emb_dim = min(config.dec_heads, 32)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim)
        self.gradient_checkpointing = False

        self.encoder = RETROEncoderBERThead(config)


    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        retrieval: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        retrieval_dates : Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if retrieval is not None:
            n, num_chunks, num_neighbors, chunk_size, device = hidden_states.shape[1], *retrieval.shape[-3:], hidden_states.device
            assert chunk_size >= self.chunk_size, 'chunk size of retrieval input must be greater or equal to the designated chunk_size on RETRO initialization'

            num_seq_chunks = n // self.chunk_size
            assert num_chunks == num_seq_chunks, f'sequence requires {num_seq_chunks} retrieved chunks, but only {num_chunks} passed in'
            if retrieval_dates:
                retrieval = retrieval_dates+retrieval

            retrieval = self.wte(retrieval)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        retrieved_encoded = False
        cross_attn_pos_emb = None
        if exists(retrieval):
            num_chunks, num_neighbors, chunk_size = retrieval.shape[-4:-1]

            cross_attn_q_pos_emb = self.rotary_pos_emb(self.chunk_size, device=device, offset=self.chunk_size - 1)  # need to add extra chunk size, since it will be shifted
            cross_attn_k_pos_emb = self.rotary_pos_emb(chunk_size, device=device)

            cross_attn_pos_emb = (cross_attn_q_pos_emb, cross_attn_k_pos_emb)
        # handle masks for encoder and decoder, if needed
        encoder_retrieved_mask = decoder_retrieved_mask = None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                )
            else:
                outputs, retrieved_encoded = block(
                    hidden_states,
                    encoder=self.encoder,
                    retrieval=retrieval,
                    context_mask=decoder_retrieved_mask,
                    encoder_retrieved_mask=encoder_retrieved_mask,
                    retrieved_encoded=retrieved_encoded,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    cross_attn_pos_emb=cross_attn_pos_emb
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


# Copied from transformers GPT
class RETROForCausalLM(RETROPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"h\.\d+\.attn\.masked_bias",
        r"lm_head.weight",
        r"h\.\d+\.attn\.attention\.bias",
    ]
    _keys_to_ignore_on_save = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = RetroModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        retrieval: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels=input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            retrieval=retrieval,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Compute loss in fp32 to match with mesh-tf version
            # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )

