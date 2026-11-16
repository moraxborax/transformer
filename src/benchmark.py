import argparse
import math
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.inference import (
    beam_search,
    default_device,
    encode_source,
    find_checkpoint,
    greedy_decode_batch,
    ids_to_tokens,
    load_model,
    special_ids,
    synchronize,
)
from src.train import collate, load_wmt14


def ngrams(tokens: list[str], size: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


def corpus_bleu(hypotheses: list[list[str]], references: list[list[str]], max_n: int = 4) -> float:
    clipped = [0] * max_n
    total = [0] * max_n
    hypothesis_length = 0
    reference_length = 0
    for hypothesis, reference in zip(hypotheses, references, strict=True):
        hypothesis_length += len(hypothesis)
        reference_length += len(reference)
        for size in range(1, max_n + 1):
            hypothesis_counts = ngrams(hypothesis, size)
            reference_counts = ngrams(reference, size)
            total[size - 1] += sum(hypothesis_counts.values())
            clipped[size - 1] += sum(min(count, reference_counts[gram]) for gram, count in hypothesis_counts.items())
    if hypothesis_length == 0 or min(total) == 0:
        return 0.0
    precisions = [clipped[index] / total[index] if total[index] else 0.0 for index in range(max_n)]
    if min(precisions) == 0.0:
        return 0.0
    brevity = 1.0 if hypothesis_length > reference_length else math.exp(1.0 - reference_length / hypothesis_length)
    return brevity * math.exp(sum(math.log(precision) for precision in precisions) / max_n) * 100.0


@torch.no_grad()
def evaluate_perplexity(
    encoder,
    decoder,
    tokenizer: Tokenizer,
    rows,
    max_seq_len: int,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    pad_id, _, _ = special_ids(tokenizer)
    total_loss = 0.0
    total_tokens = 0
    pending: list[dict[str, str]] = []
    for row in rows:
        pending.append(row["translation"])
        if len(pending) < batch_size:
            continue
        loss, count = _perplexity_batch(encoder, decoder, tokenizer, pending, max_seq_len, device, pad_id)
        total_loss += loss
        total_tokens += count
        pending = []
    if pending:
        loss, count = _perplexity_batch(encoder, decoder, tokenizer, pending, max_seq_len, device, pad_id)
        total_loss += loss
        total_tokens += count
    if total_tokens == 0:
        return float("inf"), float("inf")
    average = total_loss / total_tokens
    perplexity = math.exp(average) if average < 100 else float("inf")
    return average, perplexity


def _perplexity_batch(encoder, decoder, tokenizer, pairs, max_seq_len, device, pad_id) -> tuple[float, int]:
    collated = collate(pairs, tokenizer, max_seq_len)
    if collated is None:
        return 0.0, 0
    source, source_padding, target_in, target_padding, target_out = (tensor.to(device) for tensor in collated)
    memory = encoder(source, source_padding)
    logits = decoder(target_in, memory, target_padding, source_padding)
    token_count = int((target_out != pad_id).sum())
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_out.reshape(-1),
        ignore_index=pad_id,
        reduction="sum",
    )
    return float(loss), token_count


def reference_tokens(tokenizer: Tokenizer, text: str, max_seq_len: int) -> list[str]:
    ids = tokenizer.encode(text).ids[: max_seq_len - 1]
    tokens = []
    for token_id in ids:
        token = tokenizer.id_to_token(token_id)
        if token is not None:
            tokens.append(token)
    return tokens


@torch.no_grad()
def generate_translations(
    encoder,
    decoder,
    tokenizer: Tokenizer,
    pairs: list[dict[str, str]],
    max_seq_len: int,
    device: torch.device,
    beam_size: int,
    length_alpha: float,
) -> tuple[list[list[str]], list[list[str]], float, int]:
    pad_id, bos_id, eos_id = special_ids(tokenizer)
    hypotheses: list[list[str]] = []
    references = [reference_tokens(tokenizer, pair["en"], max_seq_len) for pair in pairs]
    generated_tokens = 0
    synchronize(device)
    started = time.perf_counter()
    if beam_size <= 1:
        source_rows = [encode_source(pair["de"], tokenizer, max_seq_len) for pair in pairs]
        width = max((len(row) for row in source_rows), default=0)
        if width == 0:
            return hypotheses, references, 0.0, 0
        source = torch.full((len(source_rows), width), pad_id, dtype=torch.long, device=device)
        padding = torch.ones(len(source_rows), width, dtype=torch.bool, device=device)
        for index, row in enumerate(source_rows):
            if not row:
                continue
            source[index, : len(row)] = torch.tensor(row, device=device)
            padding[index, : len(row)] = False
        tokens = greedy_decode_batch(encoder, decoder, source, padding, bos_id, eos_id, max_seq_len)
        for row in tokens.tolist():
            hypothesis = ids_to_tokens(tokenizer, row, bos_id, eos_id, pad_id)
            hypotheses.append(hypothesis)
            generated_tokens += len(hypothesis)
    else:
        for pair in pairs:
            source_ids = encode_source(pair["de"], tokenizer, max_seq_len)
            if not source_ids:
                hypotheses.append([])
                continue
            token_ids = beam_search(
                encoder,
                decoder,
                source_ids,
                bos_id,
                eos_id,
                max_seq_len,
                device,
                beam_size,
                length_alpha,
            )
            hypothesis = ids_to_tokens(tokenizer, token_ids, bos_id, eos_id, pad_id)
            hypotheses.append(hypothesis)
            generated_tokens += len(hypothesis)
    synchronize(device)
    elapsed = time.perf_counter() - started
    return hypotheses, references, elapsed, generated_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure perplexity, BLEU, and generation speed.")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint file; default is the latest step_*.pt")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--max-samples", type=int, default=200, help="0 uses the whole split")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--print-examples", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint or find_checkpoint(args.checkpoint_dir)
    device = torch.device(args.device)
    encoder, decoder, _checkpoint = load_model(checkpoint_path, args.n_heads, device)
    tokenizer = Tokenizer.from_file(str(args.checkpoint_dir / "tokenizer.json"))
    max_seq_len = int(encoder.position.pe.shape[0])
    dataset = load_wmt14()
    rows = list(dataset[args.split])
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    pairs = [row["translation"] for row in rows]

    average_loss, perplexity = evaluate_perplexity(
        encoder,
        decoder,
        tokenizer,
        rows,
        max_seq_len,
        device,
        args.batch_size,
    )
    hypotheses, references, elapsed, generated_tokens = generate_translations(
        encoder,
        decoder,
        tokenizer,
        pairs,
        max_seq_len,
        device,
        args.beam_size,
        args.length_penalty,
    )
    bleu = corpus_bleu(hypotheses, references)
    tokens_per_second = generated_tokens / elapsed if elapsed else 0.0
    milliseconds = (elapsed / len(pairs) * 1000.0) if pairs else 0.0

    print(f"checkpoint {checkpoint_path}")
    print(f"split {args.split}  samples {len(pairs)}  beam {args.beam_size}")
    print(f"loss {average_loss:.4f}")
    print(f"perplexity {perplexity:.2f}")
    print(f"bleu {bleu:.2f}")
    print(f"tokens/sec {tokens_per_second:.1f}")
    print(f"ms/sentence {milliseconds:.1f}")
    for pair, hypothesis in zip(pairs[: args.print_examples], hypotheses[: args.print_examples], strict=False):
        print(f"source:     {pair['de']}")
        print(f"prediction: {' '.join(hypothesis)}")
        print(f"reference:  {pair['en']}")
        print()


if __name__ == "__main__":
    main()
