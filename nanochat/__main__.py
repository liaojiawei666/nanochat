"""Training, generation and model inspection; settings live in config.json."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from nanochat.config import Config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m nanochat")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "generate", "inspect"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path)
        if name == "train":
            command.add_argument("--data-dir", type=Path, help="覆盖 config.json 中的数据目录")
        elif name == "generate":
            command.add_argument("--prompt", required=True)
            command.add_argument("--max-tokens", type=int, default=32)
            command.add_argument("--temperature", type=float, default=0.)
    args = parser.parse_args(argv)
    cfg = Config.load(args.config)
    if args.command == "train" and args.data_dir is not None:
        cfg.storage.data_dir = str(args.data_dir)
    if args.command == "train":
        from nanochat.train import train
        train(cfg)
    elif args.command == "inspect":
        import torch
        from nanochat.config import derive_training
        from nanochat.gpt import GPT, resource_estimate
        with torch.device("meta"):
            model = GPT(cfg.model)
        print(json.dumps(dict(model=vars(cfg.model), recipe=derive_training(model, cfg.train),
                              resources=resource_estimate(model)), indent=2))
    elif args.command == "generate":
        from nanochat.eval import load_model
        from nanochat.gpt import resolve_runtime, autocast_context
        device, dtype = resolve_runtime(cfg.train.dtype)
        model, tokenizer = load_model(cfg.train.output_dir, device)
        with autocast_context(device, dtype):
            tokens = model.generate(tokenizer.encode(args.prompt, add_bos=True),
                                    max_tokens=args.max_tokens, temperature=args.temperature)
        print(tokenizer.decode(tokens))


if __name__ == "__main__":
    main()
