import argparse
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.model import TransformerDecoder, TransformerEncoder
from src.train import BOS, EOS, PAD, load_wmt14

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


def length_penalty(length: int, alpha: float) -> float:
    # ((5 + |Y|) / 6) ^ alpha, from the paper's beam search.
    return ((5.0 + max(length, 1)) / 6.0) ** alpha


def find_checkpoint(directory: Path) -> Path:
    files = list(directory.glob("step_*.pt"))
    if not files:
        raise SystemExit(f"no checkpoint in {directory}")
    return max(files, key=lambda path: int(path.stem.split("_", 1)[1]))


def model_config(checkpoint: dict, n_heads: int) -> dict[str, int]:
    encoder = checkpoint["encoder"]
    vocab_size, d_model = encoder["embed.weight"].shape
    max_seq_len = encoder["position.pe"].shape[0]
    d_ff = encoder["layers.0.linear1.weight"].shape[0]
    n_layers = sum(1 for key in encoder if key.endswith(".norm1.weight") and key.startswith("layers."))
    return {
        "n_vocab": int(vocab_size),
        "d_model": int(d_model),
        "max_seq_len": int(max_seq_len),
        "d_ff": int(d_ff),
        "n_layers": int(n_layers),
        "n_heads": n_heads,
    }


def load_model(
    checkpoint_path: Path,
    n_heads: int,
    device: torch.device,
) -> tuple[TransformerEncoder, TransformerDecoder, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = model_config(checkpoint, n_heads)
    encoder = TransformerEncoder(
        config["d_model"],
        config["n_heads"],
        config["n_layers"],
        config["n_vocab"],
        config["max_seq_len"],
        config["d_ff"],
        dropout=0.0,
    )
    decoder = TransformerDecoder(
        config["d_model"],
        config["n_heads"],
        config["n_layers"],
        config["n_vocab"],
        config["max_seq_len"],
        config["d_ff"],
        dropout=0.0,
    )
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.to(device).eval()
    decoder.to(device).eval()
    return encoder, decoder, checkpoint


def special_ids(tokenizer: Tokenizer) -> tuple[int, int, int]:
    pad_id = tokenizer.token_to_id(PAD)
    bos_id = tokenizer.token_to_id(BOS)
    eos_id = tokenizer.token_to_id(EOS)
    if pad_id is None or bos_id is None or eos_id is None:
        raise SystemExit("tokenizer is missing PAD, BOS, or EOS")
    return pad_id, bos_id, eos_id


def encode_source(text: str, tokenizer: Tokenizer, max_seq_len: int) -> list[int]:
    return tokenizer.encode(text).ids[:max_seq_len]


def ids_to_text(tokenizer: Tokenizer, ids: list[int], bos_id: int, eos_id: int, pad_id: int) -> str:
    if ids and ids[0] == bos_id:
        ids = ids[1:]
    if eos_id in ids:
        ids = ids[: ids.index(eos_id)]
    ids = [token_id for token_id in ids if token_id != pad_id]
    if not ids:
        return ""
    return tokenizer.decode(ids)


def ids_to_tokens(tokenizer: Tokenizer, ids: list[int], bos_id: int, eos_id: int, pad_id: int) -> list[str]:
    if ids and ids[0] == bos_id:
        ids = ids[1:]
    if eos_id in ids:
        ids = ids[: ids.index(eos_id)]
    tokens = []
    for token_id in ids:
        if token_id == pad_id:
            continue
        token = tokenizer.id_to_token(token_id)
        if token is not None:
            tokens.append(token)
    return tokens


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def greedy_decode_batch(
    encoder: TransformerEncoder,
    decoder: TransformerDecoder,
    source: torch.Tensor,
    source_padding: torch.Tensor | None,
    bos_id: int,
    eos_id: int,
    max_seq_len: int,
) -> torch.Tensor:
    memory = encoder(source, source_padding)
    if source_padding is None:
        source_lengths = torch.full((source.size(0),), source.size(1), device=source.device)
    else:
        source_lengths = (~source_padding).sum(dim=1)
    limits = torch.clamp(source_lengths + 50, max=max_seq_len)
    tokens = torch.full((source.size(0), 1), bos_id, dtype=torch.long, device=source.device)
    finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
    for _ in range(max_seq_len - 1):
        if bool(finished.all()):
            break
        logits = decoder(tokens, memory, memory_padding_mask=source_padding)
        next_id = logits[:, -1].argmax(dim=-1)
        too_long = tokens.size(1) >= limits
        next_id = torch.where(finished | too_long, torch.full_like(next_id, eos_id), next_id)
        tokens = torch.cat([tokens, next_id.unsqueeze(1)], dim=1)
        finished = finished | too_long | (next_id == eos_id)
    return tokens


@torch.no_grad()
def greedy_decode(
    encoder: TransformerEncoder,
    decoder: TransformerDecoder,
    source_ids: list[int],
    bos_id: int,
    eos_id: int,
    max_seq_len: int,
    device: torch.device,
) -> list[int]:
    source = torch.tensor([source_ids], dtype=torch.long, device=device)
    tokens = greedy_decode_batch(encoder, decoder, source, None, bos_id, eos_id, max_seq_len)
    return tokens[0].tolist()


@torch.no_grad()
def beam_search(
    encoder: TransformerEncoder,
    decoder: TransformerDecoder,
    source_ids: list[int],
    bos_id: int,
    eos_id: int,
    max_seq_len: int,
    device: torch.device,
    beam_size: int,
    length_alpha: float,
) -> list[int]:
    source = torch.tensor([source_ids], dtype=torch.long, device=device)
    memory = encoder(source)
    limit = min(max_seq_len, len(source_ids) + 50)
    beams = [(0.0, [bos_id])]
    finished: list[tuple[float, list[int]]] = []
    for _ in range(limit - 1):
        active = [(score, tokens) for score, tokens in beams if tokens[-1] != eos_id]
        finished.extend((score, tokens) for score, tokens in beams if tokens[-1] == eos_id)
        if not active:
            break
        token_batch = torch.tensor([tokens for _, tokens in active], dtype=torch.long, device=device)
        expanded = memory.expand(token_batch.size(0), -1, -1)
        logits = decoder(token_batch, expanded)
        log_probs = logits[:, -1].log_softmax(dim=-1)
        top_scores, top_ids = log_probs.topk(beam_size, dim=-1)
        candidates: list[tuple[float, list[int]]] = []
        for row, (score, tokens) in enumerate(active):
            for rank in range(beam_size):
                token_id = int(top_ids[row, rank])
                candidates.append((score + float(top_scores[row, rank]), [*tokens, token_id]))
        candidates.sort(
            key=lambda item: item[0] / length_penalty(len(item[1]) - 1, length_alpha),
            reverse=True,
        )
        beams = candidates[:beam_size]
        if finished and all(tokens[-1] == eos_id for _, tokens in beams):
            finished.extend(beams)
            break
    pool = finished or beams
    best_score, best_tokens = max(
        pool,
        key=lambda item: item[0] / length_penalty(len(item[1]) - 1, length_alpha),
    )
    del best_score
    return best_tokens


def translate(
    encoder: TransformerEncoder,
    decoder: TransformerDecoder,
    tokenizer: Tokenizer,
    text: str,
    max_seq_len: int,
    device: torch.device,
    beam_size: int,
    length_alpha: float,
) -> str:
    pad_id, bos_id, eos_id = special_ids(tokenizer)
    source_ids = encode_source(text, tokenizer, max_seq_len)
    if not source_ids:
        return ""
    if beam_size <= 1:
        token_ids = greedy_decode(encoder, decoder, source_ids, bos_id, eos_id, max_seq_len, device)
    else:
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
    return ids_to_text(tokenizer, token_ids, bos_id, eos_id, pad_id)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate German to English with a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint file; default is the latest step_*.pt")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--beam-size", type=int, default=4, help="1 is greedy; 4 matches the paper")
    parser.add_argument("--length-penalty", type=float, default=0.6, help="beam score length penalty")
    parser.add_argument("--samples", type=int, default=0, help="translate this many WMT14 test pairs")
    parser.add_argument("--text", help="one German sentence to translate")
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint or find_checkpoint(args.checkpoint_dir)
    device = torch.device(args.device)
    encoder, decoder, _checkpoint = load_model(checkpoint_path, args.n_heads, device)
    tokenizer = Tokenizer.from_file(str(args.checkpoint_dir / "tokenizer.json"))
    max_seq_len = encoder.position.pe.shape[0]
    print(f"checkpoint {checkpoint_path} on {device}", flush=True)

    if args.text is not None:
        print(translate(encoder, decoder, tokenizer, args.text, max_seq_len, device, args.beam_size, args.length_penalty))
        return
    if args.samples > 0:
        dataset = load_wmt14()
        for index, row in enumerate(dataset["test"]):
            if index >= args.samples:
                break
            pair = row["translation"]
            prediction = translate(
                encoder,
                decoder,
                tokenizer,
                pair["de"],
                max_seq_len,
                device,
                args.beam_size,
                args.length_penalty,
            )
            print(f"source:     {pair['de']}")
            print(f"prediction: {prediction}")
            print(f"reference:  {pair['en']}")
            print(flush=True)
        return

    print("enter a German sentence, or an empty line to stop", flush=True)
    while True:
        try:
            text = input("> ")
        except EOFError:
            break
        if not text.strip():
            break
        print(translate(encoder, decoder, tokenizer, text, max_seq_len, device, args.beam_size, args.length_penalty))


if __name__ == "__main__":
    main()
