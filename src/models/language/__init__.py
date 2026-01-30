"""Language decoders."""

from src.models.language.decoders import LanguageDecoderRegistry, build_language_decoder

__all__ = [
    "LanguageDecoderRegistry",
    "build_language_decoder",
]
