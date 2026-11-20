# HypeLLM

HypeLLM runs large language models on a small graphics card, faster than
[AirLLM](https://github.com/lyogavin/airllm). It is built on top of AirLLM and replaces the part
that decides when each layer of the model is loaded and removed.

On an 8 GB card, Qwen2.5-7B went from 32 seconds per token with AirLLM to 0.09 seconds per token.

## How it works

A model is a stack of layers, and each layer has to be on the graphics card at the moment it runs.
AirLLM loads one layer from disk, runs it, removes it, and repeats that for every layer and every
token. HypeLLM changes six things:

| | AirLLM | HypeLLM |
|---|---|---|
| Layers on the graphics card | one at a time, removed after use | as many as fit stay there for good |
| Layers in RAM | none kept | the next ones are cached, so later tokens skip the disk |
| Reading from disk | one layer ahead, off when compression is on | several layers ahead, also with compression |
| Memory cleanup | after every layer (about 80 ms each) | only if memory actually runs out |
| 4-bit compression | packs every tensor, which breaks some models | packs only the large weight matrices |
| Draft model | not available | a small model guesses several tokens and the large model checks them in one pass |

Models that AirLLM handles in special ways (per-expert streaming, pre-quantized checkpoints, tables
kept on the CPU) fall back to plain AirLLM behaviour.

## Results

Measured on one PC: RTX 3060 Ti (8 GB), 32 GB RAM, SATA SSD, Windows 11. Model:
`Qwen/Qwen2.5-7B-Instruct`. Draft model: `Qwen/Qwen2.5-0.5B-Instruct`. Greedy decoding, 24 new
tokens, one short prompt.

| Setup | Seconds per token | Output |
|---|---|---|
| AirLLM, uncompressed | 32.3 | good |
| AirLLM, 4-bit | 5.4 | garbage |
| HypeLLM, uncompressed | 10 to 21 (depends on free RAM) | same tokens as AirLLM |
| HypeLLM, uncompressed, draft model | 3.6 | same tokens as without the draft |
| HypeLLM, 4-bit | 0.26 | good |
| HypeLLM, 4-bit, draft model | 0.09 | same tokens as without the draft |

Why AirLLM's 4-bit output is garbage on this model: it also packs the attention biases. Perplexity
on a short passage is 3.72 uncompressed, 55 with AirLLM's 4-bit files and 3.94 with HypeLLM's.

These numbers come from a single prompt on a single machine. Nothing larger than 7B has been
tested yet.

## Install

Python 3.10 or newer and an NVIDIA card are assumed.

```
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install "git+https://github.com/lyogavin/airllm.git#subdirectory=air_llm"
pip install transformers bitsandbytes psutil
```

Then download this repository and run the scripts from its folder. HypeLLM was developed against
AirLLM 4.0.0 (main branch, November 2026) and uses its internals, so other versions may not work.

## Use

Chat:

```
python chat.py
python chat.py --prompt "Why is the sky blue?"
```

The first run downloads the model (about 15 GB for Qwen2.5-7B, plus 1 GB for the draft model) and
writes the layer files (about 5 GB in 4-bit).

In your own code:

```python
from hypellm import AutoModel

model = AutoModel.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    compression="4bit",                        # or None
    draft_model="Qwen/Qwen2.5-0.5B-Instruct",  # optional
)
ids = model.tokenizer(["The three largest planets are"], return_tensors="pt",
                      return_attention_mask=False)["input_ids"].to(model.device)
out = model.generate(ids, max_new_tokens=24, do_sample=False)
print(model.tokenizer.decode(out[0], skip_special_tokens=True))
```

Options of `AutoModel.from_pretrained`, on top of AirLLM's own:

| Option | Meaning |
|---|---|
| `vram_budget_gb` | GB of layers kept on the graphics card (default: what is free, minus a reserve) |
| `ram_budget_gb` | GB of layers cached in RAM (default: what is free, minus a reserve) |
| `prefetch_gb` | GB the disk read-ahead may hold |
| `draft_model` | repo id or loaded model used to guess tokens; must share the main model's tokenizer |
| `draft_tokens`, `draft_confidence` | how many tokens to guess per pass, and when to stop guessing |
| `cleanup` | `'never'` (default), `'sweep'` or `'layer'` (AirLLM's behaviour) |

## Files

| File | What it is |
|---|---|
| `hypellm/engine.py` | the engine |
| `chat.py` | chat in the terminal |
| `bench.py` | times AirLLM against HypeLLM and checks that the outputs match |
| `peek.py` | prints the tokens of a sentence and the layer files on disk |

## Limits

- Tested on TinyLlama 1.1B and Qwen2.5-7B only.
- 4-bit output is close to the uncompressed model but not identical.
- The draft model only helps when it comes from the same family as the main model, and one prompt
  at a time.
- Speed depends on how much RAM and graphics memory is free when the model loads.

## Credit and license

HypeLLM depends on and adapts code from [AirLLM](https://github.com/lyogavin/airllm) by Gavin Li,
which is licensed under Apache-2.0. HypeLLM is released under the same license. See `NOTICE`.
