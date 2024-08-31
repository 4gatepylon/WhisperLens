import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import einops

# TODO(Adriano) add jaxtyping support

from transformer_lens.hook_points import HookedRootModule, HookPoint

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

        # Hook points, meant to roughly emulate what we have here:
        # https://github.com/TransformerLensOrg/TransformerLens/blob/cb5017ad0f30cde0d3ac0b0f863c27fbec964c28/transformer_lens/components/abstract_attention.py#L107C9-L113C76
        # NOTE: block has the input into attn. so we don't include a hook_point here
        self.hook_k = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_q = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_v = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_z = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_attn_scores = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_attn_scores_masked = (
            HookPoint()
        )  # [batch, head_index, query_pos, key_pos]
        self.hook_attn_pattern = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_attn_result = HookPoint()  # [batch, pos, head_index, d_model], OUTPUT

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        q = self.hook_q(q)
        k = self.hook_k(k)
        v = self.hook_v(v)
        # Attn. Scores. and Patt. Hooks inside
        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) # No scale

        qk = self.hook_attn_scores(q @ k)  # batch n_head seq seq
        if mask is not None:
            # Mask is a sq. (seq seq) tensor that normally (look below to block(..., mask=...))
            # Sets some parts to negative infinity so that they won't be included in the softmax
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()
        qk = self.hook_attn_scores_masked(qk)  # batch n_head seq seq

        # batch n_head seq seq (or for cross seq1 seq2)
        w = self.hook_attn_pattern(F.softmax(qk, dim=-1).to(q.dtype))
        return self.hook_attn_result((w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        # Hook points:
        # 1. Attn
        self.hook_resid_pre = HookPoint()  # INPUT
        self.hook_attn_ln_post = HookPoint()  # What goes into attn.
        self.hook_resid_mid = HookPoint()  # Right after adding in attention
        # 2. X-attn
        self.hook_x_attn_ln_post = HookPoint()  # What goes into x-attn.
        self.hook_x_resid_mid = (
            HookPoint()
        )  # Right after adding in cross-attention IF there is cross-attention
        # 3. MLP
        self.hook_mlp_ln_post = HookPoint()  # What goes into mlp.
        self.hook_mlp_up_post = HookPoint()  # After up-projetion from ML, before act.
        self.hook_mlp_act_post = HookPoint()  # After activation, before down-proj.
        self.hook_mlp_down_post = (
            HookPoint()
        )  # After down-proj. before adding to resid.
        # ...
        self.hook_resid_post = HookPoint()  # OUTPUT

        # MLP
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            # left to right processing; NOTE that the hook on the incoming layer-normed value is called OUTSIDE;
            # NOTE that the sequential is actually not used BECAUSE otherwise it would make loading the module
            # work in a non-desireable manner.
            Linear(n_state, n_mlp),
            # self.hook_mlp_up_post,
            nn.GELU(),
            # self.hook_mlp_act_post,
            Linear(n_mlp, n_state),
            # self.hook_mlp_down_post,
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        incoming_resid: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = self.hook_resid_pre(incoming_resid)
        x = self.hook_resid_mid(
            x
            + self.attn(
                self.hook_attn_ln_post(self.attn_ln(x)), mask=mask, kv_cache=kv_cache
            )[0]
        )
        # NOTE that cross-attn uses NO mask (so in some sense it has "listened ahead")
        if self.cross_attn:
            x = self.hook_x_resid_mid(
                x
                + self.cross_attn(
                    self.hook_x_attn_ln_post(self.cross_attn_ln(x)),
                    xa,
                    kv_cache=kv_cache,
                )[0]
            )
        mlp_in = self.hook_mlp_ln_post(self.mlp_ln(x))
        assert len(self.mlp) == 3
        assert isinstance(self.mlp[0], Linear)
        assert isinstance(self.mlp[1], nn.GELU)
        assert isinstance(self.mlp[2], Linear)
        mlp_up = self.hook_mlp_up_post(self.mlp[0](mlp_in))
        mlp_act = self.hook_mlp_act_post(self.mlp[1](mlp_up))
        mlp_down = self.hook_mlp_down_post(self.mlp[2](mlp_act))
        x = self.hook_resid_post(x + mlp_down)
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

        # Hook points:
        # 1. Before and after each convolutional layer/gelu (i.e. between each conv and act)
        self.hook_conv1_pre = HookPoint()  # INPUT
        self.hook_conv1_post_pre_act = HookPoint()
        self.hook_conv2_pre = HookPoint()
        self.hook_conv2_post_pre_act = HookPoint()
        self.hook_conv2_post_post_act = HookPoint()
        # 2. Right after above and adding the positional embedding
        self.hook_post_pos_embd_add = HookPoint()
        # 3. Before and after each block (covered by the hook points inside the blocks themselves)
        # 4. Right after the last layer norm
        self.hook_ln_post_post = HookPoint()  # OUTPUT

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.hook_conv1_post_pre_act(self.conv1(self.hook_conv1_pre(x))))
        x = F.gelu(self.hook_conv2_post_pre_act(self.conv2(self.hook_conv2_pre(x))))
        x = self.hook_conv2_post_post_act(x)
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)
        x = self.hook_post_pos_embd_add(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        x = self.hook_ln_post_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

        # Hook points:
        # 1. Embeddings
        self.hook_tokens = HookPoint()  # INPUT
        self.hook_token_embd = HookPoint()
        self.hook_pos_embd = HookPoint()
        # 2. Blocks handled per-block
        # 3. Layernorm and unembedding
        self.hook_ln_post = HookPoint()
        self.hook_logits = HookPoint()  # OUTPUT

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = self.hook_tokens(x)
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.hook_ln_post(self.ln(x))
        logits = self.hook_logits(
            (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()
        )

        return logits


class Whisper(HookedRootModule):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        # NOTE: INPUT/OUTPUT hooks are handled by the AudioEncoder and TextDecoder classes (look above)
        # (and therefore, we do not include hooks here)
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        # use the last half among the decoder layers for time alignment by default;
        # to use a specific set of heads, see `set_alignment_heads()` below.
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

        # Gives each module a parameter with its name (relative to this root module)
        # Needed for HookPoints to work
        # https://github.com/TransformerLensOrg/TransformerLens/blob/cb5017ad0f30cde0d3ac0b0f863c27fbec964c28/transformer_lens/HookedTransformer.py#L216C9-L218C21
        self.setup()

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
