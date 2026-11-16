import argparse
import os
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.utils.tensorboard import SummaryWriter

from src.model import TransformerDecoder, TransformerEncoder

PAD = "[PAD]"
BOS = "[BOS]"
EOS = "[EOS]"
UNK = "[UNK]"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]


def learning_rate(step: int, d_model: int, warmup_steps: int) -> float:
    # lrate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)
    step = max(step, 1)
    return d_model**-0.5 * min(step**-0.5, step * warmup_steps**-1.5)


def load_cached_wmt14() -> DatasetDict | None:
    import datasets

    root = Path(datasets.config.HF_DATASETS_CACHE) / "wmt___wmt14" / "de-en"
    versions = [path for path in root.glob("*/*") if path.is_dir()]
    if not versions:
        return None
    folder = max(versions, key=lambda path: path.stat().st_mtime)
    train_files = sorted(folder.glob("wmt14-train-*.arrow"))
    if not train_files:
        return None
    splits: dict[str, Dataset] = {
        "train": concatenate_datasets([Dataset.from_file(str(path)) for path in train_files])
    }
    for split in ("validation", "test"):
        path = folder / f"wmt14-{split}.arrow"
        if path.exists():
            splits[split] = Dataset.from_file(str(path))
    return DatasetDict(splits)


def load_wmt14() -> DatasetDict:
    cached = load_cached_wmt14()
    if cached is not None:
        print("using the cached wmt/wmt14 de-en files", flush=True)
        return cached
    print("downloading wmt/wmt14 de-en", flush=True)
    return load_dataset("wmt/wmt14", "de-en")


def translation_texts(rows) -> Iterator[str]:
    for row in rows:
        pair = row["translation"]
        yield pair["de"]
        yield pair["en"]


def build_tokenizer(rows, vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=UNK))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(translation_texts(rows), trainer=trainer)
    return tokenizer


def load_cached_tokenizer(path: Path, vocab_size: int) -> Tokenizer | None:
    if not path.exists():
        return None
    tokenizer = Tokenizer.from_file(str(path))
    if tokenizer.get_vocab_size() != vocab_size:
        return None
    if any(tokenizer.token_to_id(token) is None for token in SPECIAL_TOKENS):
        return None
    return tokenizer


def tie_embeddings(encoder: TransformerEncoder, decoder: TransformerDecoder) -> None:
    decoder.embed.weight = encoder.embed.weight
    decoder.output.weight = encoder.embed.weight


def unique_parameters(*modules: torch.nn.Module) -> list[torch.nn.Parameter]:
    seen: set[int] = set()
    parameters: list[torch.nn.Parameter] = []
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
    return parameters


def encode_pair(
    pair: dict[str, str],
    tokenizer: Tokenizer,
    max_seq_len: int,
    bos_id: int,
    eos_id: int,
) -> tuple[list[int], list[int], list[int]] | None:
    source = tokenizer.encode(pair["de"]).ids[:max_seq_len]
    target = tokenizer.encode(pair["en"]).ids[: max_seq_len - 1]
    if not source or not target:
        return None
    return source, [bos_id, *target], [*target, eos_id]


def pad_sequences(sequences: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(sequence) for sequence in sequences)
    tokens = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    padding = torch.ones(len(sequences), width, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        tokens[row, : len(sequence)] = torch.tensor(sequence)
        padding[row, : len(sequence)] = False
    return tokens, padding


def collate(
    pairs: list[dict[str, str]],
    tokenizer: Tokenizer,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    pad_id = tokenizer.token_to_id(PAD)
    bos_id = tokenizer.token_to_id(BOS)
    eos_id = tokenizer.token_to_id(EOS)
    encoded = [
        item
        for pair in pairs
        if (item := encode_pair(pair, tokenizer, max_seq_len, bos_id, eos_id)) is not None
    ]
    if not encoded:
        return None
    source, source_padding = pad_sequences([item[0] for item in encoded], pad_id)
    target_in, target_padding = pad_sequences([item[1] for item in encoded], pad_id)
    target_out, _ = pad_sequences([item[2] for item in encoded], pad_id)
    return source, source_padding, target_in, target_padding, target_out


def batches(rows, batch_size: int) -> Iterator[list[dict[str, str]]]:
    pending: list[dict[str, str]] = []
    while True:
        for row in rows:
            pending.append(row["translation"])
            if len(pending) == batch_size:
                yield pending
                pending = []


def train_step(
    encoder: TransformerEncoder,
    decoder: TransformerDecoder,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    pad_id: int,
    step: int,
    warmup_steps: int,
) -> torch.Tensor:
    source, source_padding, target_in, target_padding, target_out = batch
    rate = learning_rate(step, encoder.d_model, warmup_steps)
    for group in optimizer.param_groups:
        group["lr"] = rate
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(source.device.type, dtype=torch.bfloat16):
        memory = encoder(source, source_padding)
        logits = decoder(target_in, memory, target_padding, source_padding)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_out.reshape(-1),
        ignore_index=pad_id,
        label_smoothing=0.1,
    )
    loss.backward()
    optimizer.step()
    return loss.detach()


def save_checkpoint(
    path: Path,
    step: int,
    encoder: TransformerEncoder,
    decoder: TransformerDecoder,
    optimizer: torch.optim.Optimizer,
    tokenizer: Tokenizer,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path / "tokenizer.json"))
    torch.save(
        {
            "step": step,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path / f"step_{step}.pt",
    )


def train(args: argparse.Namespace) -> None:
    # The MPS allocator otherwise stops around its default watermark while the Mac still has free memory.
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    dataset = load_wmt14()
    train_rows = dataset["train"].shuffle(seed=42)
    checkpoint_dir = Path(args.checkpoint_dir)
    tokenizer_file = checkpoint_dir / "tokenizer.json"
    tokenizer = load_cached_tokenizer(tokenizer_file, args.vocab_size)
    if tokenizer is None:
        print(f"training a shared BPE vocabulary of {args.vocab_size} tokens", flush=True)
        tokenizer = build_tokenizer(train_rows, args.vocab_size)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_file))
    else:
        print(f"loading tokenizer from {tokenizer_file}", flush=True)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id(PAD)
    device = torch.device(args.device)
    encoder = TransformerEncoder(
        args.d_model,
        args.n_heads,
        args.n_layers,
        vocab_size,
        args.max_seq_len,
        args.d_ff,
        args.dropout,
    )
    decoder = TransformerDecoder(
        args.d_model,
        args.n_heads,
        args.n_layers,
        vocab_size,
        args.max_seq_len,
        args.d_ff,
        args.dropout,
    )
    tie_embeddings(encoder, decoder)
    encoder.to(device)
    decoder.to(device)
    encoder.train()
    decoder.train()
    optimizer = torch.optim.Adam(
        unique_parameters(encoder, decoder),
        lr=learning_rate(1, args.d_model, args.warmup_steps),
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    writer = SummaryWriter(checkpoint_dir / "tensorboard")
    step = 0
    try:
        for pairs in batches(train_rows, args.batch_size):
            collated = collate(pairs, tokenizer, args.max_seq_len)
            if collated is None:
                continue
            step += 1
            batch = tuple(tensor.to(device) for tensor in collated)
            loss = train_step(
                encoder,
                decoder,
                optimizer,
                batch,
                pad_id,
                step,
                args.warmup_steps,
            )
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                rate = optimizer.param_groups[0]["lr"]
                loss_value = loss.item()
                print(f"step {step} loss {loss_value:.4f} lr {rate:.6e}", flush=True)
                writer.add_scalar("loss", loss_value, step)
                writer.add_scalar("lr", rate, step)
            if step % args.save_every == 0:
                save_checkpoint(checkpoint_dir, step, encoder, decoder, optimizer, tokenizer)
            if step >= args.steps:
                break
        save_checkpoint(checkpoint_dir, step, encoder, decoder, optimizer, tokenizer)
    finally:
        writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the transformer on WMT14 English-German.")
    parser.add_argument("--steps", type=int, default=100_000, help="number of optimizer steps")
    parser.add_argument("--batch-size", type=int, default=32, help="sentence pairs per step")
    parser.add_argument("--vocab-size", type=int, default=37_000, help="shared byte-pair vocabulary size")
    parser.add_argument("--max-seq-len", type=int, default=256, help="truncate each sentence to this many tokens")
    parser.add_argument("--d-model", type=int, default=512, help="token vector width")
    parser.add_argument("--n-heads", type=int, default=8, help="attention heads per layer")
    parser.add_argument("--n-layers", type=int, default=6, help="encoder layers and decoder layers")
    parser.add_argument("--d-ff", type=int, default=2048, help="feed-forward inner width")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--warmup-steps", type=int, default=4000, help="steps before the learning rate decays")
    parser.add_argument("--log-every", type=int, default=50, help="print and record loss every this many steps")
    parser.add_argument("--save-every", type=int, default=5000, help="write a checkpoint every this many steps")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="directory for checkpoints and TensorBoard events")
    if torch.cuda.is_available():
        default_device = "cuda"
    elif torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"
    parser.add_argument("--device", default=default_device, help="torch device, for example cpu, mps, or cuda")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
