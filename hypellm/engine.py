"""HypeLLM engine: tiered layer residency on top of AirLLM's layer streaming.

Built on AirLLM (https://github.com/lyogavin/airllm, Apache-2.0, by Gavin Li). AirLLM loads every
layer from disk for every token and evicts it straight after. HypeLLM keeps AirLLM's machinery
(checkpoint splitting, the empty model on the meta device, the forward hooks) and changes what the
hooks do with each layer:

* GPU tier  - as many layers as fit in VRAM are loaded once and never evicted.
* RAM tier  - the next layers are kept in host memory (pinned where possible, still compressed if
              compression is on), so later tokens skip the disk entirely.
* Disk tier - whatever is left is streamed as in AirLLM, but a read-ahead thread keeps several
              layers in flight, wraps around into the next token, and also works with compression
              (the thread reads the packed bytes; they are expanded on the GPU when the layer runs).
"""
import bisect
import contextlib
import gc
import importlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

import airllm.utils as air_utils
from airllm.persist import ModelPersister
from airllm.utils import clean_memory

GB = 1024 ** 3


def _is_oom(err):
    oom_type = getattr(torch, 'OutOfMemoryError', None) or torch.cuda.OutOfMemoryError
    return isinstance(err, oom_type) or 'out of memory' in str(err).lower()


# ---- compression ------------------------------------------------------------------------------
#
# AirLLM packs every tensor of every shard. HypeLLM packs only the weight matrices of the decoder
# layers and the word table. Measured on Qwen2.5-7B (perplexity of a short passage, plain model
# 3.72): AirLLM's rule gives 55, and packing the attention biases alone accounts for it; this rule
# gives 3.94. Norm weights and biases are tiny, so leaving them plain costs nothing, and the output
# head stays plain because packing it costs accuracy (4.11). Shards therefore mix packed and plain
# tensors and live in their own folder (`splitted_model.hype.4bit`).

def _packable(name, value, prefixes):
    return value.ndim >= 2 and name.startswith(prefixes)


def compress_layer(state_dict, compression, prefixes):
    if compression not in ('4bit', '8bit'):
        return state_dict
    import bitsandbytes as bnb
    out = {}
    for name, value in state_dict.items():
        if not _packable(name, value, prefixes):
            out[name] = value
        elif compression == '4bit':
            packed, state = bnb.functional.quantize_nf4(value.cuda(), blocksize=64)
            out[name] = packed
            for key, item in air_utils.save_quant_state_to_dict(state).items():
                out[f'{name}.4bit.{key}'] = item
        else:
            packed, state = bnb.functional.quantize_blockwise(value.cuda(), blocksize=2048)
            out[name] = packed
            out[name + '.8bit.absmax'] = state.absmax.clone().contiguous()
            out[name + '.8bit.code'] = state.code.clone().contiguous()
    return out


def uncompress_layer(state_dict):
    """Expand a shard on the GPU. Reads AirLLM's shards too; plain tensors pass through."""
    if not any('.4bit.' in k or '.8bit.' in k for k in state_dict):
        return state_dict
    import bitsandbytes as bnb
    out = {}
    for name, value in state_dict.items():
        if '.4bit.' in name or '.8bit.' in name:
            continue
        if name + '.8bit.absmax' in state_dict:
            state = bnb.functional.QuantState(absmax=state_dict[name + '.8bit.absmax'].cuda(),
                                              code=state_dict[name + '.8bit.code'].cuda(),
                                              blocksize=2048, dtype=torch.float16)
            out[name] = bnb.functional.dequantize_blockwise(value.cuda(), state)
            continue
        prefix = name + '.4bit'
        quant = {k[len(name):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if quant:
            state = bnb.functional.QuantState.from_dict(qs_dict=quant, device='cuda')
            out[name] = bnb.functional.dequantize_nf4(value.cuda(), state)
        else:
            out[name] = value
    return out


# AirLLM's own loader (used for modules HypeLLM does not manage) must understand mixed shards too.
air_utils.uncompress_layer_state_dict = uncompress_layer


@contextlib.contextmanager
def _hype_shards():
    """While active, AirLLM's splitter writes HypeLLM-format compressed shards."""
    split, compress = air_utils.split_and_save_layers, air_utils.compress_layer_state_dict
    names = {'layer_prefix': 'model.layers', 'embed': 'model.embed_tokens'}

    def hype_split(checkpoint_path, layer_shards_saving_path=None,
                   splitted_model_dir_name='splitted_model', compression=None, layer_names=None,
                   **kwargs):
        if compression is not None:
            splitted_model_dir_name += '.hype'
            names.update(layer_names or {})
        return split(checkpoint_path, layer_shards_saving_path, splitted_model_dir_name,
                     compression=compression, layer_names=layer_names, **kwargs)

    def hype_compress(state_dict, compression=None):
        prefixes = (names['layer_prefix'] + '.', names['embed'] + '.')
        return compress_layer(state_dict, compression, prefixes)

    air_utils.split_and_save_layers = hype_split
    air_utils.compress_layer_state_dict = hype_compress
    try:
        yield
    finally:
        air_utils.split_and_save_layers = split
        air_utils.compress_layer_state_dict = compress


class HypeMixin:
    """Mix in front of any AirLLM model class to replace its per-layer load/evict policy."""

    def __init__(self, *args, vram_budget_gb=None, ram_budget_gb=None, prefetch_gb=None,
                 pin_gb=None, cleanup='never', prefetching=True, **kwargs):
        """
        vram_budget_gb : GB of layer weights to keep resident on the GPU. Default: free VRAM minus
                         room for one streamed layer and activations.
        ram_budget_gb  : GB of layer weights to keep cached in host RAM. Default: available RAM
                         minus a safety reserve.
        prefetch_gb    : GB the disk read-ahead may hold in flight. Default: two layers, min 1GB.
        pin_gb         : GB of the RAM tier that may be page-locked. Default: a quarter of RAM.
        cleanup        : when to force a garbage collection: 'never' (default; a full collection
                         costs ~80ms and memory is only reclaimed on an out-of-memory retry),
                         'sweep' (once per token) or 'layer' (AirLLM's behaviour).
        prefetching    : read ahead from disk on a worker thread.
        """
        if cleanup not in ('sweep', 'layer', 'never'):
            raise ValueError("cleanup must be 'sweep', 'layer' or 'never'")
        self._hype_opts = dict(vram_budget_gb=vram_budget_gb, ram_budget_gb=ram_budget_gb,
                               prefetch_gb=prefetch_gb, pin_gb=pin_gb)
        self._cleanup = cleanup
        self._hype_enabled = False
        # HypeLLM runs its own read-ahead, so AirLLM's one-layer prefetch stays off.
        kwargs['prefetching'] = False
        with _hype_shards():
            super().__init__(*args, **kwargs)
        self._hype_setup(prefetching)

    # ---- planning ---------------------------------------------------------------------------

    def _hype_setup(self, prefetching):
        reasons = []
        if not hasattr(self, '_streamed_indices'):
            reasons.append('streaming hooks are not installed')
        if getattr(self, '_expert_streaming', False):
            reasons.append('per-expert streaming is active')
        if self.hf_quantizer is not None:
            reasons.append('the checkpoint is pre-quantized')
        if getattr(self, '_cpu_resident_params', None):
            reasons.append('the model keeps tables on the CPU')
        if reasons:
            print(f"HypeLLM: falling back to plain AirLLM streaming because {', '.join(reasons)}.")
            self.prefetching = prefetching and self.compression is None
            if self.prefetching and self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1)
            return

        opts = self._hype_opts
        is_cuda = self.device.type == 'cuda'
        elem = torch.empty((), dtype=self.running_dtype).element_size()

        self._gpu_bytes = {}
        self._ram_bytes = {}
        for idx in self._streamed_indices:
            gpu = sum(p.numel() for p in self.layers[idx].parameters()) * elem
            shard = Path(self.checkpoint_path) / (self.layer_names[idx] + '.safetensors')
            disk = shard.stat().st_size if shard.exists() else gpu
            self._gpu_bytes[idx] = gpu
            # Compressed layers are cached still packed, so they cost their on-disk size.
            self._ram_bytes[idx] = disk if self.compression is not None else gpu

        self.hype_stats = {'wait_s': 0.0, 'place_s': 0.0, 'disk_reads': 0, 'disk_bytes': 0,
                           'sweeps': 0}
        self._stats_lock = threading.Lock()
        self._pinned = 0
        self._pin_budget = 0
        self._ram_tier = set()
        self._ram_cache = {}
        self._pending = {}
        self._pending_bytes = 0
        self._hype_executor = None
        self._last_streamed = self._streamed_indices[-1]

        # GPU tier
        largest = max(self._gpu_bytes.values())
        if opts['vram_budget_gb'] is not None:
            vram = opts['vram_budget_gb'] * GB
        elif is_cuda:
            free, total = torch.cuda.mem_get_info(self.device)
            vram = free - largest - max(1 * GB, 0.15 * total)
        else:
            vram = 0
        self._gpu_resident = set()
        used = 0
        # A resident layer saves moving its cached bytes every token, so layers that are stored
        # plain (ratio 1) go first; without compression this is simply execution order.
        def saving(idx):
            return -round(self._ram_bytes[idx] / max(self._gpu_bytes[idx], 1), 1)
        for idx in sorted(self._streamed_indices, key=lambda i: (saving(i), i)):
            if used + self._gpu_bytes[idx] <= vram:
                self._gpu_resident.add(idx)
                used += self._gpu_bytes[idx]
        self._hype_enabled = True
        self._load_gpu_tier()

        # RAM and disk tiers
        rest = [i for i in self._streamed_indices if i not in self._gpu_resident]
        largest_rest = max((self._ram_bytes[i] for i in rest), default=0)
        if opts['prefetch_gb'] is not None:
            self._prefetch_budget = opts['prefetch_gb'] * GB
        else:
            self._prefetch_budget = max(2 * largest_rest, 1 * GB)
        import psutil
        vm = psutil.virtual_memory()
        if opts['ram_budget_gb'] is not None:
            ram = opts['ram_budget_gb'] * GB
        else:
            ram = vm.available - max(4 * GB, 0.2 * vm.total) - self._prefetch_budget
        used = 0
        for idx in rest:
            if used + self._ram_bytes[idx] <= ram:
                self._ram_tier.add(idx)
                used += self._ram_bytes[idx]
        self._disk_tier = [i for i in rest if i not in self._ram_tier]
        if is_cuda:
            self._pin_budget = (opts['pin_gb'] * GB if opts['pin_gb'] is not None
                                else 0.25 * vm.total)

        # Layers that are not GPU-resident, in execution order. The read-ahead walks this cyclically.
        self._fetch_order = rest
        if prefetching and rest:
            self._hype_executor = ThreadPoolExecutor(max_workers=1)

        self.hype_plan = {
            'gpu_layers': len(self._gpu_resident),
            'gpu_gb': round(sum(self._gpu_bytes[i] for i in self._gpu_resident) / GB, 2),
            'ram_layers': len(self._ram_tier),
            'ram_gb': round(sum(self._ram_bytes[i] for i in self._ram_tier) / GB, 2),
            'disk_layers': len(self._disk_tier),
            'disk_gb': round(sum(self._ram_bytes[i] for i in self._disk_tier) / GB, 2),
        }
        p = self.hype_plan
        print(f"HypeLLM plan: {p['gpu_layers']} layers resident on GPU ({p['gpu_gb']}GB), "
              f"{p['ram_layers']} cached in RAM ({p['ram_gb']}GB), "
              f"{p['disk_layers']} streamed from disk ({p['disk_gb']}GB per token).")

    def _load_gpu_tier(self):
        """Load the GPU tier once. If VRAM runs out, the remaining layers drop to the lower tiers."""
        for idx in sorted(self._gpu_resident):
            module = self.layers[idx]
            try:
                self._hype_place(module, self._hype_read(idx))
            except RuntimeError as err:
                if not _is_oom(err):
                    raise
                module.to('meta')
                clean_memory()
                dropped = [i for i in self._gpu_resident if i >= idx]
                self._gpu_resident.difference_update(dropped)
                print(f"HypeLLM: VRAM filled up after {len(self._gpu_resident)} layers; "
                      f"{len(dropped)} planned GPU layers will be cached or streamed instead.")
                break

    # ---- reading ----------------------------------------------------------------------------

    def _hype_read(self, idx):
        """Read one layer shard into host memory. Safe to run on the read-ahead thread: no CUDA."""
        state_dict = ModelPersister.get_model_persister().load_model(
            self.layer_names[idx], self.checkpoint_path)
        nbytes = sum(v.numel() * v.element_size() for v in state_dict.values())

        if self.compression is None:
            # Cast once here instead of on every transfer to the GPU.
            for name, value in state_dict.items():
                if value.dtype != self.running_dtype and not self._should_load_verbatim(name, value):
                    state_dict[name] = value.to(self.running_dtype)

        if self._pin_budget and idx not in self._gpu_resident:
            held = sum(v.numel() * v.element_size() for v in state_dict.values())
            keep = idx in self._ram_tier
            with self._stats_lock:
                allowed = held <= self.max_pinned_layer_bytes and (
                    not keep or self._pinned + held <= self._pin_budget)
                if allowed and keep:
                    self._pinned += held
            if allowed:
                try:
                    state_dict = {k: v.pin_memory() for k, v in state_dict.items()}
                except RuntimeError:
                    if keep:
                        with self._stats_lock:
                            self._pinned -= held

        with self._stats_lock:
            self.hype_stats['disk_reads'] += 1
            self.hype_stats['disk_bytes'] += nbytes
        return state_dict

    def _hype_fetch(self, idx):
        state_dict = self._ram_cache.get(idx)
        if state_dict is not None:
            return state_dict
        start = time.perf_counter()
        future = self._pending.pop(idx, None)
        if future is not None:
            state_dict = future.result()
            self._pending_bytes -= self._ram_bytes[idx]
        else:
            state_dict = self._hype_read(idx)
        self.hype_stats['wait_s'] += time.perf_counter() - start
        if idx in self._ram_tier:
            self._ram_cache[idx] = state_dict
        return state_dict

    def _hype_schedule(self, idx):
        """Queue reads for the layers that run after `idx`, up to the read-ahead budget."""
        if self._hype_executor is None:
            return
        order = self._fetch_order
        n = len(order)
        pos = bisect.bisect_right(order, idx)
        for step in range(n):
            nxt = order[(pos + step) % n]
            if nxt == idx or nxt in self._ram_cache or nxt in self._pending:
                continue
            size = self._ram_bytes[nxt]
            if self._pending and self._pending_bytes + size > self._prefetch_budget:
                break
            self._pending[nxt] = self._hype_executor.submit(self._hype_read, nxt)
            self._pending_bytes += size

    def _hype_place(self, module, state_dict):
        start = time.perf_counter()
        for attempt in (0, 1):
            try:
                expanded = state_dict
                if self.compression is not None:
                    # Packed bytes cross to the GPU and are expanded there.
                    expanded = uncompress_layer(state_dict)
                module._airllm_moved = self.move_layer_to_device(expanded)
                break
            except RuntimeError as err:
                if attempt or not _is_oom(err):
                    raise
                # Cached-but-unused allocator blocks can fragment a nearly full GPU; release them.
                module.to('meta')
                clean_memory()
        self.hype_stats['place_s'] += time.perf_counter() - start

    # ---- hooks ------------------------------------------------------------------------------

    def _pre_hook(self, module, args):
        if not self._hype_enabled:
            return super()._pre_hook(module, args)
        idx = module._airllm_idx
        if idx in self._gpu_resident:
            self._hype_schedule(idx)
            return
        state_dict = self._hype_fetch(idx)
        # Start the next reads before spending time on the transfer.
        self._hype_schedule(idx)
        self._hype_place(module, state_dict)

    def _post_hook(self, module, args, output):
        if not self._hype_enabled:
            return super()._post_hook(module, args, output)
        idx = module._airllm_idx
        if idx not in self._gpu_resident:
            module.to('meta')
            if self._cleanup == 'layer':
                clean_memory()
        if idx == self._last_streamed:
            self.hype_stats['sweeps'] += 1
            if self._cleanup == 'sweep':
                gc.collect()
        return output

    # ---- draft model (speculative decoding) -------------------------------------------------

    def attach_draft(self, draft, draft_tokens=None, draft_confidence=None):
        """Let a small resident model guess several tokens that one sweep of this model then checks.

        Each sweep keeps the guesses that match what this model would have produced, plus one token
        of its own, so greedy output is unchanged while fewer sweeps are needed. The guessing and
        checking loop is transformers' assisted generation; HypeLLM only picks its settings.
        """
        # Model families often share a tokenizer but pad the embedding table differently (Qwen2.5
        # 0.5B has 151936 rows, 7B has 152064). transformers pairs models by vocab size, so grow
        # the draft's table to match; the added rows are zero and never win an argmax.
        main_vocab = self.config.get_text_config().vocab_size
        draft_vocab = draft.config.get_text_config().vocab_size
        if draft_vocab > main_vocab:
            raise ValueError(f"The draft model's vocabulary ({draft_vocab}) is larger than the main "
                             f"model's ({main_vocab}); they do not share a tokenizer.")
        if draft_vocab < main_vocab:
            draft.resize_token_embeddings(main_vocab, mean_resizing=False)
            with torch.no_grad():
                draft.get_output_embeddings().weight[draft_vocab:].zero_()

        streaming = not self._hype_enabled or bool(self._disk_tier)
        if draft_tokens is None:
            draft_tokens = 24 if streaming else 10
        if draft_confidence is None:
            # A sweep that reads from disk costs far more than a wasted guess, so never stop early.
            draft_confidence = 0.0 if streaming else 0.4
        config = draft.generation_config
        config.num_assistant_tokens = draft_tokens
        config.num_assistant_tokens_schedule = 'constant'
        config.assistant_confidence_threshold = draft_confidence
        self.draft = draft
        print(f"HypeLLM draft: up to {draft_tokens} guessed tokens per sweep "
              f"(stop guessing below confidence {draft_confidence}).")

    def generate(self, *args, **kwargs):
        draft = getattr(self, 'draft', None)
        if draft is not None and 'assistant_model' not in kwargs:
            kwargs['assistant_model'] = draft
        return super().generate(*args, **kwargs)


def _load_draft(repo_id, device=None, hf_token=None):
    from transformers import AutoModelForCausalLM
    from airllm.utils import default_device

    token = {'token': hf_token} if hf_token else {}
    draft = AutoModelForCausalLM.from_pretrained(repo_id, dtype='auto', **token)
    return draft.to(device or default_device()).eval()


class AutoModel:
    """Drop-in for airllm.AutoModel that returns the HypeLLM version of the matching class."""

    def __init__(self):
        raise EnvironmentError("Use AutoModel.from_pretrained(model_path_or_repo_id).")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        from airllm.auto_model import AutoModel as AirAutoModel

        lookup = {'hf_token': kwargs['hf_token']} if kwargs.get('hf_token') else {}
        module_name, class_name = AirAutoModel.get_module_class(pretrained_model_name_or_path, **lookup)
        base = getattr(importlib.import_module(module_name), class_name)
        hype_class = type('Hype' + class_name, (HypeMixin, base), {})

        # draft_model: repo id or loaded model sharing this model's tokenizer. It is loaded first
        # so the VRAM plan for the big model sees what is left.
        draft = kwargs.pop('draft_model', None)
        draft_tokens = kwargs.pop('draft_tokens', None)
        draft_confidence = kwargs.pop('draft_confidence', None)
        if isinstance(draft, str):
            draft = _load_draft(draft, kwargs.get('device'), kwargs.get('hf_token'))

        model = hype_class(pretrained_model_name_or_path, *inputs, **kwargs)
        if draft is not None:
            model.attach_draft(draft, draft_tokens, draft_confidence)
        return model
