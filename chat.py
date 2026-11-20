"""Chat with a model through HypeLLM.

    python chat.py                                  # Qwen2.5-7B in 4-bit with a 0.5B draft model
    python chat.py --prompt "Why is the sky blue?"  # answer one question and exit
    python chat.py --model <repo> --draft none --compression none

The first run downloads the model from Hugging Face and splits it into layer files.
"""
import argparse
import time

from transformers import TextStreamer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--draft', default='Qwen/Qwen2.5-0.5B-Instruct',
                   help="small model from the same family, or 'none'")
    p.add_argument('--compression', choices=['4bit', '8bit', 'none'], default='4bit')
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--prompt', default=None, help='answer one question and exit')
    args = p.parse_args()

    from hypellm import AutoModel
    model = AutoModel.from_pretrained(
        args.model,
        compression=None if args.compression == 'none' else args.compression,
        draft_model=None if args.draft == 'none' else args.draft)
    tokenizer = model.tokenizer
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    messages = []

    def answer(question):
        messages.append({'role': 'user', 'content': question})
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        ids = tokenizer([text], return_tensors='pt',
                        return_attention_mask=False)['input_ids'].to(model.device)
        start = time.perf_counter()
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                             streamer=streamer)
        elapsed = time.perf_counter() - start
        new = out[0, ids.shape[1]:]
        messages.append({'role': 'assistant',
                         'content': tokenizer.decode(new, skip_special_tokens=True)})
        print(f"[{len(new)} tokens, {len(new) / elapsed:.1f} tokens/s]\n")

    if args.prompt:
        answer(args.prompt)
        return
    print("Type a message and press Enter. An empty line or 'exit' quits.\n")
    while True:
        try:
            question = input('You: ').strip()
        except EOFError:
            break
        if not question or question.lower() in ('exit', 'quit'):
            break
        print('Model: ', end='', flush=True)
        answer(question)


if __name__ == '__main__':
    main()
