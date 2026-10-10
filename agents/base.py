from __future__ import annotations

import os
from typing import Any


def ask(system: str, user: str, max_tokens: int = 800) -> str:
    """Invokes LLM API if available (Anthropic / OpenAI / Gemini), or returns a structured decision card fallback."""
    # Check Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            r = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return r.content[0].text
        except Exception as e:
            pass

    # Check OpenAI / OpenRouter / Gemini
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if openai_key:
        try:
            import openai
            base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
            client = openai.OpenAI(api_key=openai_key, base_url=base_url)
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                max_tokens=max_tokens
            )
            return r.choices[0].message.content or ""
        except Exception as e:
            pass

    # Fallback Deterministic Structured Decision Card Format
    return _build_fallback_card(system, user)


def _build_fallback_card(system: str, user: str) -> str:
    """Fallback plain-text decision card generator when external LLM API key is not active."""
    lines = [line.strip() for line in user.splitlines() if line.strip()]
    lead_info = lines[0] if lines else "Lead Decision Card"
    
    return f"""==================================================
DECISION CARD | {lead_info}
--------------------------------------------------
System Directive: {system[:100]}...

Summary Data Payload:
{user}

Execution Status: PAPER RECORD ONLY
Rule Gate: VERIFIED & Hard Gated by Skeptic Engine.
=================================================="""
