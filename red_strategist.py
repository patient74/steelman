"""
red_strategist.py
==================

Phase 3 of the Mastercard Innovation Challenge 2026 submission:
the strategic (LLM-driven) layer of the "Generate" pillar,
sitting ON TOP of the Phase 2 bandit (red_bandit.py).

Division of labor (locked in during planning):
  - THIS module decides WHICH family (or combination of
    families, across related entities) to try each generation,
    and WHY - in natural language, logged verbatim. This is the
    part an LLM is actually good at: qualitative pattern
    reading over a short history table ("blue caught anything
    above a 0.6 mimicry blend last round, try lower this time"),
    and it is the visible "AI agent reasoning" artifact for the
    docx/live demo.
  - red_bandit.py decides the exact numeric parameter VALUES
    within whichever family the LLM picked. Bandits are
    actually good at fine-grained numeric optimization from a
    reward signal; LLMs reading a table of floats and picking
    the next float are not (this was an explicit decision from
    the planning discussion, not an oversight).

Backend: any OpenAI-compatible chat-completions endpoint (this
is what Ollama, vLLM, LM Studio, Groq, and OpenRouter all speak),
so the same code works whether the model behind it is a free,
locally-hosted open-weight model (e.g. Llama 3.1 8B / Qwen2.5 via
Ollama - the intended target here per project scope: free and
open source, zero marginal API cost) or a hosted free-tier
endpoint. Point `base_url` at whichever backend you're running.

This module ships with a MockLLMClient so the strategist's
parsing/logging/integration logic can be fully tested WITHOUT a
running model - actual inference against a local Ollama server
is expected to be run on your machine, not in this sandboxed
session (no GPU here, and Ollama needs a persistent background
server this environment can't host).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


# ============================================================
# Attack family descriptions shown to the LLM.
#
# Kept separate from payment_environment_v24.py's own
# docstrings/comments because the LLM-facing description needs
# to be a plain-English summary of WHAT THE ATTACK DOES AND WHY
# AN ATTACKER WOULD USE IT, not implementation detail - the
# strategist should reason about attacker intent, not simulator
# internals.
# ============================================================

FAMILY_DESCRIPTIONS = {
    "BEHAVIORAL_MIMICRY": (
        "Blends fraud into a specific customer's own historical "
        "spending pattern (amount, timing, merchant choice). "
        "Higher mimicry_blend = more convincing impersonation of "
        "the real customer, harder for point-in-time rules to "
        "catch."
    ),
    "TEMPORAL_MIMICRY": (
        "Only the transaction TIMING is shifted to match the "
        "victim's usual hour-of-day pattern. A narrower, cheaper "
        "version of behavioral mimicry - useful for isolating "
        "whether timing alone is enough to evade detection."
    ),
    "AMOUNT_MIMICRY": (
        "Only the transaction AMOUNT is shifted to match the "
        "victim's own historical amount distribution. Isolates "
        "amount-based evasion from timing/merchant evasion."
    ),
    "ADAPTIVE_CARD_TESTING": (
        "A card-testing campaign that reacts to the outcome of "
        "its own previous attempt within the same campaign: "
        "escalates amount after a successful charge, retreats "
        "and rotates merchant/MCC after a decline. Models an "
        "attacker (or automated bot) that adapts in real time."
    ),
    "RELATIONAL_CAMOUFLAGE": (
        "Spreads fraud thinly across many entities (customers/"
        "devices/merchants) instead of concentrating it, so no "
        "single transaction looks anomalous on its own - only "
        "visible via shared-entity graph structure. "
        "spread_density controls how thin the spread is."
    ),
    "AGENT_IDENTITY_SPOOFING": (
        "A fraudulent transaction claims to come from a "
        "legitimate customer's payment agent, but its "
        "behavioural signature (amount scale) doesn't quite "
        "match that agent's real history. spoofing_similarity "
        "controls how close the forgery is."
    ),
    "AGENT_SCOPE_ABUSE": (
        "A genuine payment agent transacts outside its own "
        "declared authorization scope (spend limit and/or "
        "allowed merchant categories). scope_violation_depth "
        "controls how far outside scope; escalation_pace "
        "controls whether the violation appears abruptly or "
        "creeps in gradually across the campaign."
    ),
}


# ============================================================
# LLM client protocol - anything with this method works.
# ============================================================

class LLMClient(Protocol):

    def complete(self, system: str, user: str) -> str:
        ...


class OpenAICompatibleClient:
    """
    Thin wrapper around the `openai` SDK pointed at any
    OpenAI-compatible endpoint. For a local, free, open-source
    model via Ollama:

        client = OpenAICompatibleClient(
            base_url="http://localhost:11434/v1",
            api_key="ollama",              # unused, but required by the SDK
            model="llama3.1:8b",           # or "qwen2.5:14b-instruct" etc.
        )

    Ollama must already be running (`ollama serve`) with the
    model pulled (`ollama pull llama3.1:8b`) on YOUR machine -
    this class does not start or manage the server.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.4,
        max_tokens: int = 600,
        min_seconds_between_calls: float = 0.0,
        max_retries: int = 5,
    ):

        from openai import OpenAI

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_seconds_between_calls = (
            min_seconds_between_calls
        )
        self.max_retries = max_retries
        self._last_call_time = 0.0

    def complete(self, system: str, user: str) -> str:

        import time as _time

        if self.min_seconds_between_calls > 0:

            elapsed = (
                _time.time() - self._last_call_time
            )

            wait = (
                self.min_seconds_between_calls - elapsed
            )

            if wait > 0:
                _time.sleep(wait)

        last_error = None

        for attempt in range(self.max_retries):

            try:

                response = (
                    self._client.chat.completions.create(
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        messages=[
                            {
                                "role": "system",
                                "content": system,
                            },
                            {"role": "user", "content": user},
                        ],
                    )
                )

                self._last_call_time = _time.time()

                return response.choices[0].message.content

            except Exception as e:

                last_error = e

                is_rate_limit = (
                    "429" in str(e)
                    or "rate" in str(e).lower()
                    or "quota" in str(e).lower()
                )

                if (
                    is_rate_limit
                    and attempt < self.max_retries - 1
                ):

                    backoff = min(
                        2 ** attempt * 2, 60
                    )

                    print(
                        f"    [rate limited, retrying in "
                        f"{backoff}s ... attempt "
                        f"{attempt + 1}/{self.max_retries}]"
                    )

                    _time.sleep(backoff)

                    continue

                raise

        raise last_error


class MockLLMClient:
    """
    Deterministic stand-in for an LLM, used to test
    RedStrategist's prompt construction, response parsing, and
    logging WITHOUT a running model. Cycles through a fixed
    script of responses (or repeats the last one once exhausted)
    so integration tests are fully reproducible.
    """

    def __init__(self, scripted_responses: list[str]):

        self.scripted_responses = scripted_responses
        self.call_count = 0
        self.calls_log: list[dict] = []

    def complete(self, system: str, user: str) -> str:

        self.calls_log.append(
            {"system": system, "user": user}
        )

        idx = min(
            self.call_count, len(self.scripted_responses) - 1
        )

        self.call_count += 1

        return self.scripted_responses[idx]


# ============================================================
# Strategist
# ============================================================

RESPONSE_SCHEMA_INSTRUCTIONS = """
Respond with ONLY a JSON object (no markdown fences, no other \
text) in exactly this shape:

{
  "reasoning": "2-4 sentences explaining your choice, grounded \
in the detection history you were given",
  "actions": [
    {"family": "<one of the listed family names>", "intensity": "low" | "medium" | "high"}
  ]
}

"actions" may contain 1-3 entries if you want to try a \
combination of families this generation (e.g. targeting related \
entities). "intensity" is a rough qualitative hint for the \
tactical search that follows you - it does not need to be \
precise.
""".strip()


@dataclass
class StrategistDecision:

    reasoning: str
    families: list[str]
    intensities: list[str]
    raw_response: str
    parse_ok: bool


class RedStrategist:
    """
    Wraps an LLMClient to produce per-generation family/combo
    choices plus a logged natural-language rationale. Falls back
    to a safe default decision if the model's response can't be
    parsed, so a flaky/small local model can never crash the
    generation loop - it just degrates to "pick least-tried
    family" for that one generation (same policy RedBandit uses
    standalone in Phase 2).
    """

    def __init__(
        self,
        client: LLMClient,
        parameterized_families: list[str],
        fallback_family_chooser,
        log_path: str | Path = "strategist_reasoning_log.jsonl",
    ):

        self.client = client
        self.parameterized_families = parameterized_families
        self.fallback_family_chooser = fallback_family_chooser
        self.log_path = Path(log_path)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------

    def _build_prompt(
        self,
        generation: int,
        history: list[dict],
        max_history: int = 12,
    ) -> tuple[str, str]:

        system = (
            "You are the red-team strategist in an adversarial "
            "payment-fraud simulation used for a security "
            "research competition. Your job is to choose which "
            "GenAI-powered fraud attack family (or short "
            "combination of families) to attempt next, based on "
            "what has and hasn't evaded the blue-team detector "
            "so far. You are optimizing for realistic, "
            "explainable attack strategy discovery - not for "
            "actually committing fraud. Be concise and specific "
            "about WHY, referencing the numbers you were given."
        )

        family_lines = []
        for f in self.parameterized_families:
            desc = FAMILY_DESCRIPTIONS.get(f, "")
            family_lines.append(f"- {f}: {desc}")

        recent = history[-max_history:]

        if recent:
            history_lines = []
            for h in recent:
                history_lines.append(
                    f"  gen={h.get('generation')} "
                    f"family={h.get('family')} "
                    f"params={h.get('params')} "
                    f"evasion_rate={h.get('evasion_rate'):.2f} "
                    f"mean_detection_margin="
                    f"{h.get('detection_margin'):+.3f}"
                )
            history_text = "\n".join(history_lines)
        else:
            history_text = (
                "  (no history yet - this is the first "
                "generation)"
            )

        user = (
            f"Generation {generation}.\n\n"
            f"Available attack families:\n"
            + "\n".join(family_lines)
            + "\n\nRecent results "
              f"(last {len(recent)} campaigns; lower/negative "
              "mean_detection_margin = harder for blue to "
              "catch = more evasive; evasion_rate is the "
              "fraction of that campaign's transactions blue "
              "did NOT flag):\n"
            + history_text
            + "\n\n"
            + RESPONSE_SCHEMA_INSTRUCTIONS
        )

        return system, user

    # --------------------------------------------------------

    def _parse_response(
        self, raw: str
    ) -> StrategistDecision:

        text = raw.strip()

        # Small models frequently wrap JSON in markdown fences
        # even when told not to - strip those defensively rather
        # than failing on something this easy to recover from.
        text = re.sub(
            r"^```(json)?|```$", "", text, flags=re.MULTILINE
        ).strip()

        try:
            parsed = json.loads(text)

            families = [
                a["family"]
                for a in parsed["actions"]
                if a.get("family")
                in self.parameterized_families
            ]

            intensities = [
                a.get("intensity", "medium")
                for a in parsed["actions"]
                if a.get("family")
                in self.parameterized_families
            ]

            if not families:
                raise ValueError(
                    "No valid families in parsed response"
                )

            return StrategistDecision(
                reasoning=parsed.get("reasoning", ""),
                families=families,
                intensities=intensities,
                raw_response=raw,
                parse_ok=True,
            )

        except Exception:

            fallback_family = self.fallback_family_chooser()

            return StrategistDecision(
                reasoning=(
                    "[FALLBACK - could not parse LLM response] "
                    "Defaulted to least-tried family policy."
                ),
                families=[fallback_family],
                intensities=["medium"],
                raw_response=raw,
                parse_ok=False,
            )

    # --------------------------------------------------------

    def decide(
        self,
        generation: int,
        history: list[dict],
    ) -> StrategistDecision:

        system, user = self._build_prompt(generation, history)

        raw = self.client.complete(system, user)

        decision = self._parse_response(raw)

        self._log(generation, system, user, decision)

        return decision

    # --------------------------------------------------------

    def _log(self, generation, system, user, decision):

        record = {
            "generation": generation,
            "prompt_user": user,
            "reasoning": decision.reasoning,
            "families": decision.families,
            "intensities": decision.intensities,
            "parse_ok": decision.parse_ok,
            "raw_response": decision.raw_response,
        }

        with open(
            self.log_path, "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(record) + "\n")

    # --------------------------------------------------------

    def generate_debrief(
        self,
        full_history: list[dict],
    ) -> str:
        """
        End-of-run summary (Idea #4 from the brainstorm): asks
        the same LLM to write a short natural-language debrief
        of what it found most evasive, for direct inclusion in
        the solution walkthrough docx. Falls back to a templated
        summary if the model can't produce usable prose (still
        useful, just less colorful).
        """

        if not full_history:
            return "No campaigns were run - nothing to debrief."

        sorted_by_evasion = sorted(
            full_history,
            key=lambda h: h.get("evasion_rate", 0),
            reverse=True,
        )

        top = sorted_by_evasion[:5]

        system = (
            "You are the red-team strategist writing a short "
            "end-of-engagement debrief for a security research "
            "report. Be specific and reference actual numbers."
        )

        top_lines = "\n".join(
            f"  family={h.get('family')} params={h.get('params')} "
            f"evasion_rate={h.get('evasion_rate'):.2f}"
            for h in top
        )

        user = (
            "Here are the 5 most evasive campaigns across the "
            "whole run:\n" + top_lines + "\n\n"
            "Write a 4-6 sentence debrief: which strategy was "
            "most evasive and at roughly what parameter values, "
            "and what that implies about a static, non-adaptive "
            "fraud defense's blind spots. Plain prose, no JSON, "
            "no markdown headers."
        )

        try:
            return self.client.complete(system, user).strip()
        except Exception as e:

            lines = [
                "[Auto-generated fallback debrief - LLM call "
                f"failed: {e}]",
                "",
                "Most evasive campaigns observed:",
            ]

            for h in top:
                lines.append(
                    f"  - {h.get('family')} "
                    f"(params={h.get('params')}): "
                    f"evasion_rate={h.get('evasion_rate'):.2f}"
                )

            return "\n".join(lines)
