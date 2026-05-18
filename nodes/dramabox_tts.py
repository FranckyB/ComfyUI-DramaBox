"""
DramaBox TTS — ComfyUI node that runs the full LTX audio diffusion pipeline.

DramaBox weights (~8.5 GB) download automatically from HuggingFace on first use into:

    ComfyUI/models/dramabox/
        dramabox-dit-v1.safetensors
        dramabox-audio-components.safetensors
        silence_latent_frame.pt

Gemma 3 12B weights are loaded from ComfyUI/models/text_encoders/ (fp8 safetensors)
or from a full HuggingFace snapshot in ComfyUI/models/dramabox/.
Existing weights in the old node-local models/ folder are migrated automatically.
Once present, files are loaded directly — no HuggingFace API calls on startup.

Inputs
------


prompt          Scene description with quoted dialogue
seed            Reproducibility seed
steps           Denoising steps (default 30)
cfg_scale       Classifier-free guidance (default 2.5)
stg_scale       Skip-token guidance (default 1.5)
voice_ref       Optional AUDIO — reference voice to clone
negative_prompt What the model should avoid
options         Optional DRAMABOX_OPTIONS from DramaBox Options node

Outputs
-------
audio           ComfyUI AUDIO: {"waveform": Tensor[B,C,S], "sample_rate": int}
info            Human-readable stats string
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

try:
    from server import PromptServer as _PromptServer
except Exception:
    _PromptServer = None

logger = logging.getLogger(__name__)

# ── ensure bundled src/ and ltx2/ are importable ─────────────────────────────
_NODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("src", "ltx2"):
    _p = os.path.join(_NODE_DIR, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .dramabox_clip import DramaBoxTextEncoderLoader  # noqa: E402

# Use ComfyUI's central models folder so weights are shared with other nodes.
# Fall back to the node-local models/ dir when running outside ComfyUI.
try:
    import folder_paths as _folder_paths
    _MODELS_DIR = _folder_paths.models_dir
except Exception:
    _MODELS_DIR = os.path.join(_NODE_DIR, "models")
os.makedirs(os.path.join(_MODELS_DIR, "dramabox"), exist_ok=True)

_DEFAULT_NEG = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "robotic voice, echo, background noise, off-sync audio, repetitive speech"
)


# ─────────────────────────────────────────────────────────────────────────────
# Model caches — keyed by str(device)
#
# _LOADED_MODELS:        audio VAE (conditioner + decoder) + transformer
# _LOADED_TEXT_ENCODER:  Gemma + embeddings processor (text encoder only)
#
# Both caches are wrapped in ModelPatcher so ComfyUI moves them GPU↔CPU
# automatically per generation stage.  The text encoder lives in a separate
# cache so an external DRAMABOX_CLIP can be used without reloading everything.
# ─────────────────────────────────────────────────────────────────────────────

_LOADED_MODELS: dict[str, dict] = {}
_LOADED_TEXT_ENCODER: dict[str, dict] = {}


_GEMMA_FP4_FILENAME  = "gemma_3_12B_it_fp4_mixed.safetensors"
_GEMMA_FP4_REPO      = "Comfy-Org/ltx-2"
_GEMMA_FP4_REPO_PATH = "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"


def _get_dramabox_setting(key, default=None):
    """Read a value from ComfyUI's persisted user settings JSON."""
    try:
        import json
        import folder_paths as _fp
        settings_path = os.path.join(_fp.base_path, "user", "default", "comfy.settings.json")
        if os.path.isfile(settings_path):
            with open(settings_path, encoding="utf-8") as _f:
                return json.load(_f).get(key, default)
    except Exception:
        pass
    return default


def _get_dramabox_bool_setting(key: str, default: bool = False) -> bool:
    """Read a boolean DramaBox user setting with tolerant coercion."""
    v = _get_dramabox_setting(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    return bool(default)


def _offload_clip_patcher_to_cpu(clip_obj) -> None:
    """Offload a CLIP patcher to CPU and remove it from loaded-GPU tracking."""
    import comfy.model_management as mm

    patcher = getattr(clip_obj, "patcher", None)
    if patcher is None:
        return

    try:
        patcher.unpatch_model(patcher.offload_device)
    except Exception as e:
        logger.debug("[DramaBox] clip patcher offload failed: %s", e)

    mm.current_loaded_models[:] = [
        m for m in mm.current_loaded_models if id(m.model) != id(patcher)
    ]
    mm.soft_empty_cache()

def _find_or_download_gemma_path():
    """Return a local path to the default Gemma text encoder.

    Resolution order:
      1. User preference set in ComfyUI Settings → DramaBox → Default Text Encoder.
      2. ``gemma_3_12B_it_fp4_mixed.safetensors`` in any ``text_encoders`` folder.
      3. Auto-download ``gemma_3_12B_it_fp4_mixed.safetensors`` from ``Comfy-Org/ltx-2``.

    To use a different model per-workflow, connect a DramaBox CLIP Loader node.
    """
    import shutil
    _fp = None
    try:
        import folder_paths as _fp
        te_dirs = _fp.get_folder_paths("text_encoders")
    except Exception:
        te_dirs = [os.path.join(_MODELS_DIR, "text_encoders")]

    gguf_dirs = []
    if _fp is not None:
        for key in ("clip_gguf", "clip"):
            try:
                gguf_dirs.extend(_fp.get_folder_paths(key))
            except Exception:
                pass

    # Keep order stable and remove duplicates.
    all_dirs = list(dict.fromkeys(te_dirs + gguf_dirs))

    # 1. User preference from ComfyUI settings
    preferred = (_get_dramabox_setting("DramaBox.defaultTextEncoder") or "").strip()
    if preferred:
        stem, ext = os.path.splitext(preferred)
        preferred_names = [preferred] if ext else [stem + ".safetensors", stem + ".gguf"]

        import glob as _glob
        for preferred_name in preferred_names:
            # Exact file-key lookup first.
            if _fp is not None:
                for key in ("text_encoders", "clip_gguf", "clip"):
                    try:
                        candidate = _fp.get_full_path(key, preferred_name)
                    except Exception:
                        candidate = None
                    if candidate and os.path.isfile(candidate):
                        return candidate

            # Fallback recursive search in known directories.
            for folder in all_dirs:
                candidate = os.path.join(folder, preferred_name)
                if os.path.isfile(candidate):
                    return candidate
                matches = _glob.glob(os.path.join(folder, "**", preferred_name), recursive=True)
                if matches:
                    return matches[0]

        print(
            f"[DramaBox] Warning: preferred text encoder '{preferred}' not found "
            "(checked safetensors/gguf) — falling back to default."
        )

    # 2. Default fp4 mixed file
    import glob as _glob
    for folder in te_dirs:
        exact = os.path.join(folder, _GEMMA_FP4_FILENAME)
        if os.path.isfile(exact):
            return exact
        matches = _glob.glob(os.path.join(folder, "**", _GEMMA_FP4_FILENAME), recursive=True)
        if matches:
            return matches[0]
    target_dir = te_dirs[0]
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, _GEMMA_FP4_FILENAME)
    print(f"[DramaBox] Gemma not found — downloading {_GEMMA_FP4_FILENAME} from {_GEMMA_FP4_REPO} …")
    from huggingface_hub import snapshot_download
    import logging as _logging
    _httpx_log = _logging.getLogger("httpx")
    _prev = _httpx_log.level
    _httpx_log.setLevel(_logging.WARNING)
    try:
        snapshot_download(
            repo_id=_GEMMA_FP4_REPO,
            allow_patterns=[_GEMMA_FP4_REPO_PATH],
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            token=os.environ.get("HF_TOKEN"),
        )
    finally:
        _httpx_log.setLevel(_prev)
    # snapshot_download places the file at local_dir/repo_path; move it flat
    dl_path = os.path.join(target_dir, _GEMMA_FP4_REPO_PATH)
    if os.path.isfile(dl_path) and os.path.abspath(dl_path) != os.path.abspath(target_path):
        shutil.move(dl_path, target_path)
    # Remove leftover subdirs created by huggingface_hub
    for leftover in ("split_files", ".cache"):
        d = os.path.join(target_dir, leftover)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    if os.path.isfile(target_path):
        print(f"[DramaBox] Downloaded: {target_path}")
        return target_path
    raise FileNotFoundError(
        f"[DramaBox] Failed to download {_GEMMA_FP4_FILENAME}. "
        "Place it manually in your ComfyUI/models/text_encoders/ folder, "
        "or connect a DramaBox CLIP Loader node to supply the text encoder."
    )


def _estimate_bytes(module: torch.nn.Module) -> int:
    """Rough VRAM estimate for a module (parameters + buffers)."""
    total = 0
    for t in (*module.parameters(), *module.buffers()):
        if not getattr(t, 'is_meta', False):
            total += t.numel() * t.element_size()
    return max(total, 1)


def _make_patcher(module: torch.nn.Module, load_device: torch.device):
    """Wrap *module* in a ComfyUI ModelPatcher so the memory manager tracks it."""
    import comfy.model_management as _mm
    import comfy.model_patcher
    return comfy.model_patcher.ModelPatcher(
        module,
        load_device=load_device,
        offload_device=_mm.unet_offload_device(),
        size=_estimate_bytes(module),
    )


class _DramaBoxTextBundle(torch.nn.Module):
    """Single nn.Module wrapping Gemma + embeddings processor.

    Treated as one unit by ComfyUI's ModelPatcher so both models move
    GPU↔CPU together as a single eviction unit — matching how LTXVideo
    wraps CLIP text encoders inside comfy.sd.CLIP.
    """

    def __init__(self, text_encoder, embeddings_processor):
        super().__init__()
        self.text_encoder = text_encoder
        self.embeddings_processor = embeddings_processor


def _load_models(device) -> dict:
    """Load the audio VAE, transformer, and audio decoder with CPU-offload.

    All components are moved to GPU per-stage via mm.load_models_gpu().
    ComfyUI's free_memory() evicts each stage's models back to CPU when the
    next stage loads, so only one large model occupies VRAM at a time.

    The Gemma text encoder is NOT loaded here — it is handled separately by
    _load_text_encoder() or supplied via a DRAMABOX_CLIP input.
    """
    import comfy.model_management as mm

    cache_key = str(device)
    if cache_key in _LOADED_MODELS:
        logger.info("[DramaBox] Using cached models.")
        return _LOADED_MODELS[cache_key]

    offload_device = mm.unet_offload_device()   # typically torch.device('cpu')
    torch_dtype = torch.bfloat16

    # -- Resolve weight paths (local-first via patched model_downloader) -------
    from model_downloader import get_model_path
    logger.info("[DramaBox] Resolving model weights…")
    ckpt_transformer = get_model_path("transformer",      cache_dir=_MODELS_DIR)
    ckpt_audio       = get_model_path("audio_components", cache_dir=_MODELS_DIR)

    from ltx_pipelines.utils.blocks import AudioConditioner, AudioDecoder

    # ── 1. AudioConditioner (VAE encoder) — load to CPU ──────────────────────
    logger.info("[DramaBox] Loading AudioConditioner (CPU)…")
    try:
        audio_conditioner = AudioConditioner(
            checkpoint_path=ckpt_audio,
            dtype=torch_dtype,
            device=offload_device,
            warm=True,
        )
    except Exception as exc:
        logger.warning("[DramaBox] CPU init failed for AudioConditioner (%s) — falling back to GPU", exc)
        audio_conditioner = AudioConditioner(
            checkpoint_path=ckpt_audio,
            dtype=torch_dtype,
            device=device,
            warm=True,
        )
    mm.soft_empty_cache()

    # ── 2. Transformer (LTX audio-only DiT) — build on CPU ───────────────────
    logger.info("[DramaBox] Loading Transformer (CPU)…")
    from safetensors import safe_open
    from ltx_core.loader.registry import DummyRegistry
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
    from ltx_core.loader.sd_ops import SDOps
    from ltx_core.model.transformer.model import LTXModel, LTXModelType
    from ltx_core.model.transformer.rope import LTXRopeType
    from ltx_core.model.transformer.text_projection import create_caption_projection
    from ltx_core.model.transformer.attention import AttentionFunction
    from ltx_core.model.model_protocol import ModelConfigurator

    with safe_open(ckpt_transformer, framework="pt") as f:
        config = json.loads(f.metadata()["config"])

    class _AudioOnlyConfigurator(ModelConfigurator[LTXModel]):
        @classmethod
        def from_config(cls, cfg):
            _t = cfg.get("transformer", {})
            cp = None
            if not _t.get("caption_proj_before_connector", False):
                with torch.device("meta"):
                    cp = create_caption_projection(_t, audio=True)
            return LTXModel(
                model_type=LTXModelType.AudioOnly,
                audio_num_attention_heads=_t.get("audio_num_attention_heads", 32),
                audio_attention_head_dim=_t.get("audio_attention_head_dim", 64),
                audio_in_channels=_t.get("audio_in_channels", 128),
                audio_out_channels=_t.get("audio_out_channels", 128),
                num_layers=_t.get("num_layers", 48),
                audio_cross_attention_dim=_t.get("audio_cross_attention_dim", 2048),
                norm_eps=_t.get("norm_eps", 1e-6),
                attention_type=AttentionFunction(_t.get("attention_type", "default")),
                positional_embedding_theta=10000.0,
                audio_positional_embedding_max_pos=[20.0],
                timestep_scale_multiplier=_t.get("timestep_scale_multiplier", 1000),
                use_middle_indices_grid=_t.get("use_middle_indices_grid", True),
                rope_type=LTXRopeType(_t.get("rope_type", "interleaved")),
                double_precision_rope=_t.get("frequencies_precision", False) == "float64",
                apply_gated_attention=_t.get("apply_gated_attention", False),
                audio_caption_projection=cp,
                cross_attention_adaln=_t.get("cross_attention_adaln", False),
            )

    sd_ops = (
        SDOps("AO")
        .with_matching(prefix="model.diffusion_model.")
        .with_replacement("model.diffusion_model.", "")
    )
    try:
        transformer = (
            Builder(
                model_path=ckpt_transformer,
                model_class_configurator=_AudioOnlyConfigurator,
                model_sd_ops=sd_ops,
                registry=DummyRegistry(),
            )
            .build(device=offload_device, dtype=torch_dtype)
            .to(offload_device)
            .eval()
        )
    except Exception as exc:
        logger.warning("[DramaBox] CPU build failed for Transformer (%s) — falling back to GPU", exc)
        transformer = (
            Builder(
                model_path=ckpt_transformer,
                model_class_configurator=_AudioOnlyConfigurator,
                model_sd_ops=sd_ops,
                registry=DummyRegistry(),
            )
            .build(device=device, dtype=torch_dtype)
            .to(device)
            .eval()
        )
    n_params = sum(p.numel() for p in transformer.parameters()) / 1e9
    logger.info("[DramaBox] Transformer: %.1fB params", n_params)
    mm.soft_empty_cache()

    # ── 3. AudioDecoder (VAE decoder + vocoder) — load to CPU ────────────────
    logger.info("[DramaBox] Loading AudioDecoder (CPU)…")
    try:
        audio_decoder = AudioDecoder(
            checkpoint_path=ckpt_audio,
            dtype=torch_dtype,
            device=offload_device,
            warm=True,
        )
    except Exception as exc:
        logger.warning("[DramaBox] CPU init failed for AudioDecoder (%s) — falling back to GPU", exc)
        audio_decoder = AudioDecoder(
            checkpoint_path=ckpt_audio,
            dtype=torch_dtype,
            device=device,
            warm=True,
        )
    mm.soft_empty_cache()

    # ── Build stage-specific patcher groups ───────────────────────────────────
    # voice_patchers : audio VAE encoder (step 3, optional)
    # xfmr_patchers  : diffusion transformer (step 7)
    # dec_patchers   : audio VAE decoder + vocoder (step 9)
    voice_patchers = [_make_patcher(audio_conditioner._warm_encoder, device)]
    xfmr_patchers  = [_make_patcher(transformer, device)]
    dec_patchers   = [
        _make_patcher(audio_decoder._warm_decoder, device),
        _make_patcher(audio_decoder._warm_vocoder, device),
    ]

    model = {
        "voice_patchers":    voice_patchers,
        "xfmr_patchers":     xfmr_patchers,
        "dec_patchers":      dec_patchers,
        "audio_conditioner": audio_conditioner,
        "transformer":       transformer,
        "audio_decoder":     audio_decoder,
        "device":            device,
        "dtype":             torch_dtype,
    }
    _LOADED_MODELS[cache_key] = model
    logger.info("[DramaBox] Non-text components loaded (audio VAE, transformer, decoder).")
    return model


def _load_text_encoder(device):
    """Auto-load the DramaBox Gemma text encoder as a standard ComfyUI CLIP.

    Uses DramaBoxTextEncoderLoader so the fallback path is identical to the
    connected-CLIP path — both go through DramaBoxTEModel + ModelPatcher.
    Result is cached by (device, path); a settings change triggers a reload.
    """
    from .dramabox_clip import DramaBoxTextEncoderLoader

    gemma_path = _find_or_download_gemma_path()
    cache_key = f"{device}:{gemma_path}"

    if cache_key in _LOADED_TEXT_ENCODER:
        return _LOADED_TEXT_ENCODER[cache_key]

    # Evict any stale entry for this device (different path)
    stale = [k for k in list(_LOADED_TEXT_ENCODER) if k.startswith(f"{device}:")]
    for k in stale:
        del _LOADED_TEXT_ENCODER[k]

    print(f"[DramaBox] Loading text encoder: {os.path.basename(gemma_path)}")
    clip_device = "default"  # follow Comfy's default text-encoder device policy
    (clip,) = DramaBoxTextEncoderLoader().load(gemma_path, clip_device)
    _LOADED_TEXT_ENCODER[cache_key] = clip
    return clip


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_lora_path(lora_name: str) -> str | None:
    """Resolve a LoRA name to an absolute path via ComfyUI's folder system."""
    import folder_paths
    path = folder_paths.get_full_path("loras", lora_name)
    if path and os.path.isfile(path):
        return path
    if os.path.isabs(lora_name) and os.path.isfile(lora_name):
        return lora_name
    return None


def _apply_lora_deltas(transformer: torch.nn.Module, lora_path: str, strength: float) -> list:
    """Apply LoRA weights to transformer in-place using manual delta math.

    Handles both PEFT format (base_model.model.*) and original ID-LoRA format
    (diffusion_model.*), matching the approach in src/inference.py.

    Returns a list of (param_name, delta_tensor) so the caller can undo the
    changes after inference by subtracting each delta.
    """
    from safetensors.torch import load_file as _st_load

    lora_sd = _st_load(lora_path)
    is_peft = any("base_model.model." in k for k in lora_sd)
    is_idlora = any("diffusion_model." in k for k in lora_sd)

    # Build {param_path: {"A": tensor, "B": tensor, "alpha": float}} mapping
    pairs: dict[str, dict] = {}
    for k, v in lora_sd.items():
        if is_peft:
            if "base_model.model." not in k:
                continue
            base = k.replace("base_model.model.", "")
            if ".lora_A." in base:
                pp = base[: base.index(".lora_A.")]
                pairs.setdefault(pp, {})["A"] = v
            elif ".lora_B." in base:
                pp = base[: base.index(".lora_B.")]
                pairs.setdefault(pp, {})["B"] = v
        elif is_idlora:
            if "diffusion_model." not in k:
                continue
            base = k.replace("diffusion_model.", "")
            if ".lora_A.weight" in base:
                pp = base.replace(".lora_A.weight", "")
                pairs.setdefault(pp, {})["A"] = v
            elif ".lora_B.weight" in base:
                pp = base.replace(".lora_B.weight", "")
                pairs.setdefault(pp, {})["B"] = v

    param_dict = dict(transformer.named_parameters())
    applied: list[tuple[str, torch.Tensor]] = []

    for pp, pair in pairs.items():
        if "A" not in pair or "B" not in pair:
            continue
        lora_A, lora_B = pair["A"], pair["B"]
        rank = lora_A.shape[0]
        scale = strength  # alpha == rank by default (scale = alpha/rank * strength = strength)

        weight_key = pp + ".weight"
        if weight_key not in param_dict:
            continue

        param = param_dict[weight_key]
        dev, dt = param.device, param.dtype

        # delta = scale * lora_B @ lora_A   (computed in float32 for precision)
        delta = scale * (
            lora_B.to(device=dev, dtype=torch.float32)
            @ lora_A.to(device=dev, dtype=torch.float32)
        ).to(dtype=dt)

        param.data.add_(delta)
        applied.append((weight_key, delta))

    return applied


def _remove_lora_deltas(transformer: torch.nn.Module, applied: list) -> None:
    """Undo deltas previously applied by _apply_lora_deltas."""
    param_dict = dict(transformer.named_parameters())
    for weight_key, delta in applied:
        if weight_key in param_dict:
            p = param_dict[weight_key]
            p.data.sub_(delta.to(device=p.device, dtype=p.dtype))


def _auto_rescale(cfg: float) -> float:
    """CFG-aware std-rescale schedule (prevents clipping at high CFG)."""
    if cfg <= 2.0:
        return 0.0
    if cfg <= 3.0:
        return 0.6 * (cfg - 2.0)
    if cfg <= 4.0:
        return 0.6 + 0.2 * (cfg - 3.0)
    if cfg <= 8.0:
        return 0.8
    return min(1.0, 0.8 + 0.1 * (cfg - 8.0))


def _collect_dramabox_patchers() -> list:
    """Return all internal DramaBox ModelPatchers currently cached."""
    all_patchers = []
    for clip in _LOADED_TEXT_ENCODER.values():
        try:
            all_patchers.append(clip.patcher)
        except AttributeError:
            pass
    for model in _LOADED_MODELS.values():
        for group in ("voice_patchers", "xfmr_patchers", "dec_patchers"):
            all_patchers.extend(model.get(group, []))
    return all_patchers


def _offload_dramabox_models_to_cpu():
    """Move cached DramaBox models to CPU but keep them in RAM for fast reuse."""
    import comfy.model_management as mm

    if not _LOADED_MODELS and not _LOADED_TEXT_ENCODER:
        logger.info("[DramaBox] No models loaded — nothing to offload.")
        return

    all_patchers = _collect_dramabox_patchers()

    for patcher in all_patchers:
        try:
            patcher.unpatch_model(patcher.offload_device)
        except Exception as e:
            logger.debug("[DramaBox] unpatch_model failed during offload: %s", e)

    patcher_set = set(id(p) for p in all_patchers)
    mm.current_loaded_models[:] = [
        m for m in mm.current_loaded_models if id(m.model) not in patcher_set
    ]

    mm.soft_empty_cache()
    logger.info("[DramaBox] Cached models offloaded to CPU (kept in RAM).")


def _unload_dramabox_models():
    """Free all DramaBox model weights from GPU and CPU RAM.

    Unpatches all ComfyUI-managed ModelPatchers (moving them to CPU), removes
    them from current_loaded_models, then clears _LOADED_MODELS so GC can
    reclaim memory.  After this call the next DramaBox TTS generation will
    reload everything from disk.
    """
    import gc
    import comfy.model_management as mm

    if not _LOADED_MODELS and not _LOADED_TEXT_ENCODER:
        logger.info("[DramaBox] No models loaded — nothing to free.")
        return

    # Collect all internally-managed patchers (external CLIP is user-managed)
    all_patchers = _collect_dramabox_patchers()

    for patcher in all_patchers:
        try:
            patcher.unpatch_model(patcher.offload_device)
        except Exception as e:
            logger.debug("[DramaBox] unpatch_model failed: %s", e)

    # ── 2. Remove patchers from ComfyUI's tracking list ──────────────────
    patcher_set = set(id(p) for p in all_patchers)
    mm.current_loaded_models[:] = [
        m for m in mm.current_loaded_models if id(m.model) not in patcher_set
    ]

    _LOADED_MODELS.clear()
    _LOADED_TEXT_ENCODER.clear()
    gc.collect()
    mm.soft_empty_cache()
    logger.info("[DramaBox] All models unloaded. VRAM freed.")


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class DramaBoxTTS:
    """
    Generate expressive, voice-clonable speech using the DramaBox LTX diffusion model.

    All weights (DramaBox + Gemma) download automatically on first use
    and are cached in ComfyUI-DramaBox/models/.
    """

    CATEGORY = "DramaBox"
    DESCRIPTION = (
        "Generate expressive TTS with optional voice cloning.\n"
        "All weights (DramaBox + Gemma) auto-download on first use.\n"
        "Optionally connect any audio source as voice_ref to clone a voice."
    )
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2**31 - 1, "control_after_generate": True},
                ),
                "use_prompt_input": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "on",
                        "label_off": "off",
                        "tooltip": "When on, use the connected prompt input instead of the text widget.",
                    },
                ),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Enter prompt...",
                        "tooltip": (
                            "Scene description with quoted dialogue.\n"
                            "Example: 'A woman speaks warmly, \"Hello there.\"'"
                        ),
                    },
                ),
            },
            "optional": {
                "voice_ref": (
                    "AUDIO",
                    {"tooltip": "Reference voice for cloning. 5–15 s of clean speech works best."},
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "forceInput": True,
                        "lazy": True,
                        "tooltip": "Connect any text source here to override the text widget.",
                    },
                ),
                "lora_stack": (
                    "LORA_STACK",
                    {"tooltip": "Optional LoRA stack (from any LoRA stacker node). Applied to the transformer only."},
                ),
                "options": (
                    "DRAMABOX_OPTIONS",
                    {"tooltip": "Connect DramaBox Options for cfg, steps, duration, and more."},
                ),
                "dramabox_clip": (
                    "CLIP",
                    {"tooltip": "Pre-loaded text encoder from DramaBox Text Encoder Loader. Skips internal Gemma loading when connected."},
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def IS_CHANGED(cls, seed, use_prompt_input=False, text="", prompt=None, **kwargs):
        return (seed, use_prompt_input, prompt)

    def check_lazy_status(self, seed, use_prompt_input=False, text="", **kwargs):
        """Request the connected prompt only when use_prompt_input is enabled."""
        return ["prompt"] if use_prompt_input else []

    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def generate(
        self,
        seed: int = 0,
        use_prompt_input: bool = False,
        text: str = "",
        prompt: str | None = None,
        voice_ref=None,
        options: dict | None = None,
        lora_stack=None,
        dramabox_clip=None,
        unique_id=None,
    ):
        import comfy.model_management as mm
        device = mm.get_torch_device()

        # Use connected text source when the toggle is on, else fall back to widget text
        used_prompt = prompt if (use_prompt_input and prompt) else text

        # Notify the frontend so it can display the active prompt in the widget
        if unique_id is not None and _PromptServer is not None:
            try:
                _PromptServer.instance.send_sync(
                    "dramabox-tts-update",
                    {"node_id": unique_id, "prompt": used_prompt, "use_prompt_input": use_prompt_input},
                )
            except Exception:
                pass

        # ── Load (or retrieve cached) model components ───────────────────
        # _load_models() builds ModelPatchers on first call; subsequent calls
        # return the cache. load_models_gpu() is called per-stage so only one
        # large model occupies VRAM at a time; ComfyUI evicts the previous
        # stage's weights to CPU automatically.
        model = _load_models(device)

        opts = options or {}
        ref_duration: float  = opts.get("ref_duration", 10.0)
        gen_duration: float  = opts.get("gen_duration", 0.0)
        duration_mult: float = opts.get("duration_multiplier", 1.1)
        speed: float         = float(opts.get("speed", 1.0))
        steps: int           = int(opts.get("steps", 30))
        cfg_scale: float     = float(opts.get("cfg_scale", 2.5))
        stg_scale: float     = float(opts.get("stg_scale", 1.5))
        negative_prompt: str = opts.get("negative_prompt", _DEFAULT_NEG)
        rescale_raw          = opts.get("rescale_scale", "auto")
        rescale_scale: float = (
            _auto_rescale(cfg_scale) if rescale_raw == "auto" else float(rescale_raw)
        )

        default_policy = "offload_to_cpu" if _get_dramabox_bool_setting("DramaBox.autoOffload", True) else "keep_loaded"
        offload_policy = str(opts.get("post_generate_model_policy", default_policy))
        if offload_policy not in {"keep_loaded", "offload_to_cpu", "offload"}:
            offload_policy = default_policy

        # ── Resolve text encoder (external CLIP or auto-loaded CLIP) ───────────
        # Both paths produce a standard comfy.sd.CLIP — same encoding code runs
        # regardless of whether the user connected a DramaBoxTextEncoderLoader.
        _clip_enc = dramabox_clip if dramabox_clip is not None else _load_text_encoder(device)

        audio_cond  = model["audio_conditioner"]
        transformer = model["transformer"]
        decoder     = model["audio_decoder"]
        torch_dtype = model["dtype"]

        # ── imports from bundled ltx2 / src ─────────────────────────────
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.components.patchifiers import AudioPatchifier
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
        from ltx_core.components.schedulers import LTX2Scheduler
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.model.transformer.model import X0Model
        from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
        from ltx_core.tools import AudioLatentTools
        from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
        from ltx_pipelines.utils.denoisers import GuidedDenoiser
        from ltx_pipelines.utils.samplers import euler_denoising_loop
        from audio_conditioning import AudioConditionByReferenceLatent
        from inference import estimate_speech_duration

        t_total = time.time()
        patchifier = AudioPatchifier(patch_size=1)

        # ── 1. Compute target shape ──────────────────────────────────────
        if gen_duration and gen_duration > 0:
            gen_dur = float(gen_duration)
        else:
            gen_dur = round(estimate_speech_duration(used_prompt, speed) * duration_mult, 1)
        fps = 25.0
        n_frames = int(round(gen_dur * fps)) + 1
        n_frames = ((n_frames - 1 + 4) // 8) * 8 + 1
        pixel_shape  = VideoPixelShape(batch=1, frames=n_frames, height=64, width=64, fps=fps)
        target_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
        audio_tools  = AudioLatentTools(patchifier=patchifier, target_shape=target_shape)
        logger.info("[DramaBox] target: %.1fs → %d frames", gen_dur, n_frames)

        # ── 2. Initial latent state ──────────────────────────────────────
        state = audio_tools.create_initial_state(device=device, dtype=torch_dtype)

        # ── 3. Voice reference conditioning ─────────────────────────────
        mm.load_models_gpu(model["voice_patchers"])
        if voice_ref is not None:
            try:
                from ltx_pipelines.utils.media_io import decode_audio_from_file
                import tempfile, soundfile as sf

                waveform = voice_ref["waveform"]          # [B, C, S]
                sr_in: int = voice_ref["sample_rate"]
                wav = waveform[0].cpu().float().numpy()   # [C, S]
                if wav.ndim == 2:
                    wav = wav.T                           # → [S, C]

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, wav, sr_in)
                    tmp_path = tmp.name

                try:
                    voice = decode_audio_from_file(tmp_path, device, 0.0, ref_duration)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

                if voice is not None:
                    w = voice.waveform
                    if w.dim() == 2:
                        w = w.repeat(2, 1) if w.shape[0] == 1 else w
                        w = w.unsqueeze(0)
                    elif w.dim() == 3 and w.shape[1] == 1:
                        w = w.repeat(1, 2, 1)
                    n_ref = int(ref_duration * voice.sampling_rate)
                    if w.shape[-1] < n_ref:
                        w = w.repeat(1, 1, (n_ref // w.shape[-1]) + 1)
                    w = w[..., :n_ref]
                    peak = w.abs().max()
                    if peak > 0:
                        w = w * (10 ** (-4.0 / 20) / peak)
                    voice = Audio(waveform=w, sampling_rate=voice.sampling_rate)
                    ref_latent = audio_cond(lambda enc: vae_encode_audio(voice, enc, None))
                    cond = AudioConditionByReferenceLatent(
                        latent=ref_latent.to(device, torch_dtype), strength=1.0
                    )
                    state = cond.apply_to(state, audio_tools)
                    logger.info("[DramaBox] Voice reference encoded.")
            except Exception as exc:
                logger.warning("[DramaBox] voice_ref failed (%s) — running without it", exc)

        # ── 4. Add noise ────────────────────────────────────────────────
        gen = torch.Generator(device=device).manual_seed(seed)
        state = GaussianNoiser(generator=gen)(state, noise_scale=1.0)

        # ── 5. Encode text prompts ───────────────────────────────────────
        # _clip_enc is always a comfy.sd.CLIP — either user-supplied or auto-loaded.
        # Load the text encoder to GPU just for encoding, then let ComfyUI evict it.
        logger.info("[DramaBox] Encoding prompts…")
        mm.load_models_gpu([_clip_enc.patcher])
        tokens_pos = _clip_enc.tokenize(used_prompt)
        cond_pos   = _clip_enc.encode_from_tokens(tokens_pos, return_dict=True)
        a_ctx      = cond_pos["cond"].to(device=device, dtype=torch_dtype)
        del cond_pos, tokens_pos
        a_ctx_neg  = None
        if cfg_scale > 1.0:
            tokens_neg = _clip_enc.tokenize(negative_prompt)
            cond_neg   = _clip_enc.encode_from_tokens(tokens_neg, return_dict=True)
            a_ctx_neg  = cond_neg["cond"].to(device=device, dtype=torch_dtype)
            del cond_neg, tokens_neg

        # Guided denoising concatenates positive/negative contexts along batch,
        # so sequence length must match. Prompt trimming may produce different
        # lengths (e.g. 256 vs 128), so pad the shorter one on the left.
        if a_ctx_neg is not None and a_ctx.dim() == 3 and a_ctx_neg.dim() == 3:
            if a_ctx.shape[1] != a_ctx_neg.shape[1]:
                target_len = max(a_ctx.shape[1], a_ctx_neg.shape[1])

                def _left_pad_ctx(ctx: torch.Tensor, tgt: int) -> torch.Tensor:
                    cur = ctx.shape[1]
                    if cur >= tgt:
                        return ctx[:, -tgt:, :]
                    pad = ctx.new_zeros((ctx.shape[0], tgt - cur, ctx.shape[2]))
                    return torch.cat((pad, ctx), dim=1)

                logger.info(
                    "[DramaBox] Aligning context lengths for CFG: pos=%d neg=%d -> %d",
                    a_ctx.shape[1], a_ctx_neg.shape[1], target_len,
                )
                a_ctx = _left_pad_ctx(a_ctx, target_len)
                a_ctx_neg = _left_pad_ctx(a_ctx_neg, target_len)

        # Unified offload policy from Options (or preference-based default).
        if offload_policy in {"offload_to_cpu", "offload"}:
            _offload_clip_patcher_to_cpu(_clip_enc)

        mm.soft_empty_cache()

        # ── 6. Build denoiser ────────────────────────────────────────────
        guider = MultiModalGuider(
            params=MultiModalGuiderParams(
                cfg_scale=cfg_scale,
                stg_scale=stg_scale,
                stg_blocks=[29],
                rescale_scale=rescale_scale,
                modality_scale=1.0,
            ),
            negative_context=a_ctx_neg,
        )
        denoiser = GuidedDenoiser(
            v_context=None,
            a_context=a_ctx,
            video_guider=None,
            audio_guider=guider,
        )

        # ── 7. Diffusion sampling ────────────────────────────────────────
        # Apply LoRAs via manual delta math (matching src/inference.py's PEFT
        # approach). Deltas are added before moving the model to GPU and
        # subtracted after inference so the cached transformer stays unmodified.
        xfmr_patchers = model["xfmr_patchers"]
        _applied_deltas: list = []  # [(weight_key, delta_tensor), …]
        if lora_stack:
            for lora_name, strength_model, _strength_clip in lora_stack:
                lora_path = _resolve_lora_path(lora_name)
                if lora_path is None:
                    logger.warning("[DramaBox] LoRA not found, skipping: %s", lora_name)
                    continue
                deltas = _apply_lora_deltas(transformer, lora_path, float(strength_model))
                _applied_deltas.extend(deltas)
                logger.info(
                    "[DramaBox] LoRA applied: %s (strength=%.2f, params=%d)",
                    os.path.basename(lora_path), strength_model, len(deltas),
                )
        mm.load_models_gpu(xfmr_patchers)
        logger.info(
            "[DramaBox] Denoising (%d steps, cfg=%.1f, stg=%.1f)…",
            steps, cfg_scale, stg_scale,
        )
        sigmas = LTX2Scheduler().execute(steps=steps, latent=state.latent).to(device)
        x0 = X0Model(transformer)
        _, audio_state = euler_denoising_loop(
            sigmas=sigmas,
            video_state=None,
            audio_state=state,
            stepper=EulerDiffusionStep(),
            transformer=x0,
            denoiser=denoiser,
        )

        mm.soft_empty_cache()

        # Restore the cached transformer by undoing any LoRA deltas applied above.
        if _applied_deltas:
            _remove_lora_deltas(transformer, _applied_deltas)
            _applied_deltas.clear()

        # ── 8. Unpatchify + end-of-clip silence-prior fix ─────────────────
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

        latent = audio_state.latent
        # The DiT bakes a silence prior at frame 513; interpolate across it
        # for longer generations to avoid an audible glitch.
        if latent.shape[2] > 513:
            patched = latent.clone()
            for f in (512, 513):
                t = (f - 511) / 3
                patched[:, :, f, :] = (
                    (1 - t) * latent[:, :, 511, :] + t * latent[:, :, 514, :]
                )
            latent = patched

        # ── 9. Decode latents → waveform ────────────────────────────────
        mm.load_models_gpu(model["dec_patchers"])
        logger.info("[DramaBox] Decoding…")
        decoded = decoder(latent)

        mm.soft_empty_cache()

        # ── 10. Return as ComfyUI AUDIO ──────────────────────────────────
        waveform = decoded.waveform.cpu().float()
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)   # [B=1, C, S]
        sample_rate = decoded.sampling_rate
        audio_dur = waveform.shape[-1] / sample_rate
        elapsed = time.time() - t_total

        if offload_policy == "offload_to_cpu":
            logger.info("[DramaBox] offload_policy=offload_to_cpu")
            _offload_dramabox_models_to_cpu()
        elif offload_policy == "offload":
            logger.info("[DramaBox] offload_policy=offload")
            _unload_dramabox_models()

        return ({"waveform": waveform, "sample_rate": sample_rate},)


# ─────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "DramaBoxTTS": DramaBoxTTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DramaBoxTTS": "DramaBox TTS",
}
