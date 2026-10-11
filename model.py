"""OpenBench SE wrapper for Rec-RIR with VEM and RIR-MINT back ends."""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoint.tar"


@dataclass(frozen=True)
class AudioResult:
    audio_path: str
    rir_path: str
    prior_path: str | None
    sample_rate: int
    mode: str

    @property
    def text(self):
        return self.audio_path


class RecRIRModel:
    def __init__(self, model_path=None, device="auto"):
        import toml
        import torch

        from acoustics.feature import transforms
        from method.pim import PIM
        from method.vem_rir_omit3bin_nb import VEM
        from network.RecRIR import BiSpatialNet

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device(device)
        config = toml.load(str(ROOT / "config.toml"))
        self.sample_rate = int(config["acoustic"]["sr"])
        self.tf = transforms(**config["acoustic"])
        self.network = BiSpatialNet(**config["model"]).to(self.device).eval()
        checkpoint = Path(model_path or DEFAULT_CHECKPOINT).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("Model checkpoint not found: {}".format(checkpoint))
        checkpoint_data = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        weights = {
            key.removeprefix("module."): value
            for key, value in checkpoint_data["model"].items()
            if not any(part in key for part in ("ops", "params"))
        }
        self.network.load_state_dict(weights, strict=True)
        self.pim = PIM(**config["pim"])
        vem_config = dict(config["vem"])
        if os.environ.get("REC_RIR_VEM_MAX_STEPS"):
            vem_config["max_steps"] = int(os.environ["REC_RIR_VEM_MAX_STEPS"])
        self.vem = VEM(**vem_config)
        self.mint_config = config["mint"]

    def run_tool(self, audio_path, output_path=None, mode="vem"):
        import librosa
        import numpy as np
        import soundfile as sf
        import torch

        from mint.dereverb_linear import mint_multichannel_inverse_linear

        if mode not in ("vem", "mint"):
            raise ValueError("mode must be 'vem' or 'mint'")

        source = Path(audio_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError("Input audio not found: {}".format(source))
        samples, rate = sf.read(str(source), dtype="float32", always_2d=True)
        if samples.shape[0] == 0:
            raise ValueError("Input audio is empty")
        mono = samples.mean(axis=1)
        if rate != self.sample_rate:
            mono = librosa.resample(mono, orig_sr=rate, target_sr=self.sample_rate)
        if not np.isfinite(mono).all():
            raise ValueError("Input audio contains non-finite samples")

        if output_path is None:
            output_dir = ROOT / ".runtime" / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            descriptor, destination = tempfile.mkstemp(
                prefix=source.stem + "_" + mode + "_", suffix=".wav", dir=str(output_dir)
            )
            os.close(descriptor)
        else:
            destination = str(Path(output_path).expanduser().resolve())
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
        stem = Path(destination).with_suffix("")
        rir_destination = str(stem) + "_rir.wav"
        prior_destination = str(stem) + "_prior.wav" if mode == "vem" else None

        if np.max(np.abs(mono)) <= 1e-12:
            enhanced = np.zeros(len(mono), dtype=np.float32)
            rir = np.zeros(self.sample_rate * 2, dtype=np.float32)
            prior = enhanced if mode == "vem" else None
        else:
            signal = torch.from_numpy(mono.copy()).to(self.device)
            with torch.no_grad():
                if mode == "vem":
                    scale = signal.abs().max()
                    normalized = (signal / scale).unsqueeze(0).unsqueeze(0)
                    observed = self.tf.stft(normalized, "complex")
                    speech_features, _, _ = self.network(self.tf.preprocess(observed))
                    prior_spectrum = self.tf.postprocess(speech_features)
                    prior_abs = prior_spectrum.abs().clamp_min(1e-8)
                    prior_audio = self.tf.istft(prior_spectrum, "complex").squeeze()
                    estimate, impulse, _, _ = self.vem.process(
                        signal.clone(), None, self.tf, str(self.device), prior_abs=prior_abs
                    )
                    enhanced = estimate.cpu().numpy()
                    prior = (prior_audio * scale).cpu().numpy()
                    rir = impulse.cpu().numpy()
                    if len(enhanced) < len(mono):
                        enhanced = np.pad(enhanced, (0, len(mono) - len(enhanced)))
                    enhanced = enhanced[: len(mono)]
                    if len(prior) < len(mono):
                        prior = np.pad(prior, (0, len(mono) - len(prior)))
                    prior = prior[: len(mono)]
                else:
                    impulse = self.pim.process(signal, self.network, self.tf, str(self.device))
                    rir = impulse.cpu().numpy()
                    taps = rir[: int(self.mint_config["rir_len"])]
                    if np.max(np.abs(taps)) <= 1e-12:
                        raise RuntimeError("Estimated RIR is silent; MINT cannot be designed")
                    delay = int(np.argmax(np.abs(taps)))
                    estimate = mint_multichannel_inverse_linear(
                        mono[:, None], [taps],
                        inverse_len=int(self.mint_config["inverse_len"]),
                        delay=delay,
                        reg=float(self.mint_config["reg"]),
                        fs=self.sample_rate,
                    )
                    enhanced = np.pad(estimate[delay:], (0, delay))[: len(mono)]
                    prior = None

            rir_peak = np.max(np.abs(rir)) if len(rir) else 0
            if rir_peak > 0:
                rir = rir / rir_peak
            if not np.isfinite(enhanced).all() or not np.isfinite(rir).all():
                raise RuntimeError("Rec-RIR produced non-finite output")
            if prior is not None and not np.isfinite(prior).all():
                raise RuntimeError("Rec-RIR speech prior is non-finite")

        sf.write(destination, enhanced, self.sample_rate, subtype="FLOAT")
        sf.write(rir_destination, rir, self.sample_rate, subtype="FLOAT")
        if prior_destination is not None:
            sf.write(prior_destination, prior, self.sample_rate, subtype="FLOAT")
        return AudioResult(destination, rir_destination, prior_destination, self.sample_rate, mode)


Model = RecRIRModel
