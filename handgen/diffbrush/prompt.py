"""The IAM checkpoint's alphabet and line geometry."""

from __future__ import annotations

IAM_LETTERS = " _!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def model_text(text: str) -> str:
    """Whitespace belongs to page layout. All other characters must be representable."""
    prompt = " ".join(text.split())
    unsupported = sorted(set(prompt) - set(IAM_LETTERS))
    if unsupported:
        raise ValueError(f"DiffBrush's IAM alphabet has no glyph for {', '.join(map(repr, unsupported))}")
    if not prompt:
        raise ValueError("DiffBrush requires a non-empty prompt")
    if len(prompt) > 500:
        raise ValueError("The DiffBrush IAM content encoder supports at most 500 characters per prompt")
    return prompt


def canvas_width(text: str) -> int:
    """Allow writing room for wide letters while limiting repeated short phrases."""
    if len(text) >= 35:
        return 1024
    return max(96, 64 * ((len(text) * 25 + 80) // 64))
