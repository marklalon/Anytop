"""Which frozen text encoder bakes joint names, species descriptors and action words.

Torch-free, so preprocessing and validation can read the defaults without loading
a model. ``model.conditioners.build_text_conditioner`` turns a name into the encoder.
"""

DEFAULT_TEXT_ENCODER = "google/embeddinggemma-300m"

# sentence-transformers encoders: the prompt they encode under and the output width.
# A name not listed here is a T5 (``T5Conditioner.MODELS``).
SENTENCE_ENCODER_SPECS = {
    "google/embeddinggemma-300m": {"prompt_name": "STS", "dim": 768},
}


def is_sentence_encoder(name: str) -> bool:
    return name in SENTENCE_ENCODER_SPECS


def text_encoder_pooling(name: str) -> tuple[str, str]:
    """``(pooling, eos_policy)`` the action word table's contract records for *name*."""
    if is_sentence_encoder(name):
        return f"sentence_transformers:{SENTENCE_ENCODER_SPECS[name]['prompt_name']}", "model"
    return "masked_mean", "keep"
