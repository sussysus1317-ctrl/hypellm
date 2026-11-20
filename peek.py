"""Show the real pieces HypeLLM works with: the tokens of a sentence and the layer files on disk."""
import glob
import os
import re
import sys

from transformers import AutoTokenizer

MODEL = 'Qwen/Qwen2.5-7B-Instruct'
text = ' '.join(sys.argv[1:]) or 'The three largest planets in the solar system are'

tokenizer = AutoTokenizer.from_pretrained(MODEL)
ids = tokenizer(text)['input_ids']
print(f'TEXT   : {text}')
print(f'TOKENS : {len(ids)} pieces')
for token_id in ids:
    print(f'   {token_id:>7}  {tokenizer.decode([token_id])!r}')

pattern = os.path.join(os.environ.get('HF_HOME', ''), 'hub', 'models--' + MODEL.replace('/', '--'),
                       'snapshots', '*', 'splitted_model.hype.4bit', '*.safetensors')
files = glob.glob(pattern)


def order(path):
    name = os.path.basename(path)
    number = re.search(r'layers\.(\d+)\.', name)
    if number:
        return 1, int(number.group(1))
    return (0, 0) if 'embed' in name else (2, 0) if 'norm' in name else (3, 0)


print(f'\nLAYER FILES : {len(files)} files, '
      f'{sum(os.path.getsize(f) for f in files) / 1024 ** 3:.1f} GB in total')
for path in sorted(files, key=order):
    print(f'   {os.path.getsize(path) / 1024 ** 2:>7.0f} MB  {os.path.basename(path)}')
