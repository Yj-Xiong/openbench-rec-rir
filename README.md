# Rec-RIR Dereverberation: OpenBench SE Skill

This independent Skill turns the Rec-RIR `epoch35` model into two usable
speech-dereverberation paths. The default path uses Rec-RIR's predicted clean
speech spectrum as a prior for VEM, similar to VINP. The alternative path
estimates an RIR with PIM and applies regularized single-channel RIR-MINT.

| Item | Value |
|---|---|
| Task | `SE` (single-channel speech dereverberation) |
| Input | Reverberant audio path, resampled to 16 kHz mono |
| Default output | Enhanced 16 kHz float WAV in `audio_path` (`mode=vem`) |
| Other outputs | `rir_path`; `prior_path` for VEM mode |
| Model | `network/RecRIR.py:BiSpatialNet` |
| Checkpoint | `checkpoint.tar`, copy of upstream `ckpt/epoch35.tar` |
| Runtime | PyTorch FP32, CUDA, batch size 1 |
| MCP tool | `enhance_speech` |

The checkpoint's 473 network keys load with `strict=True` after removal of
training-only entries, matching the upstream `inference.py` loader.

## Modes

- `vem` (default): predict the direct-path speech complex spectrum, use its
  magnitude as the VEM prior, and return VEM-enhanced speech, prior speech and
  VEM's RIR estimate. The default is 100 VEM steps.
- `mint`: estimate the RIR with Rec-RIR's original PIM, then use the
  regularized linear-convolution MINT implementation from
  `bci_mint_scripts/mint/dereverb_linear.py`. The uploaded copy is
  `mint/dereverb_linear.py`. It uses the first 1024 RIR taps, a 512-tap
  inverse filter and regularization `0.001` by default.

Both modes preserve the 16 kHz input duration in their `audio_path` output.
MINT is a single-channel regularized inverse in this package and may amplify
noise when the RIR estimate is poor. Compare modes on the same evaluation
split; do not interpret the RIR waveform as enhanced speech.

## Install and Validate

Use Python 3.10+ with a compatible CUDA PyTorch/`torchaudio` build.

```bash
cd Rec-RIR/openbench_rec_rir
bash setup.sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python validate.py
```

`validate.py` checks both MCP modes with the real checkpoint. It uses three
VEM steps for a quick smoke test; production inference uses 100. Set
`REC_RIR_VEM_MAX_STEPS` only when intentionally changing this protocol.

## MCP Contract

The stdio MCP server supports `initialize`, `tools/list`, `tools/call` and
`ping`. Call `enhance_speech` with `audio_path`, optional `output_path`, and
optional `mode` (`vem` or `mint`). The result has `audio_path`, `rir_path`,
`sample_rate=16000`, and `mode`; VEM adds `prior_path`. Both
`structuredContent` and text content carry the paths. Without an output path,
files are written to `.runtime/outputs/`.

## Source and License

Rec-RIR model, PIM and checkpoint originate from [Rec-RIR](https://github.com/Audio-WestlakeU/Rec-RIR).
VEM follows the VINP implementation with an explicit external prior input.
The included [MIT license](LICENSE) covers the Rec-RIR source snapshot;
verify the license of the separately sourced MINT implementation and the
checkpoint redistribution terms before public upload.
