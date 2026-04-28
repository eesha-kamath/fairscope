"""
gemini_client.py
Handles all calls to the Gemini API via Google AI Studio REST endpoint.
Includes retry logic with exponential backoff for rate limit handling.
"""

import os
import json
import time
import requests
from typing import Optional

_HARDCODED_API_KEY = ""


def get_api_key() -> Optional[str]:
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        return key
    return _HARDCODED_API_KEY if _HARDCODED_API_KEY else None


def call_gemini(prompt: str, system_instruction: str = "", temperature: float = 0.3, retries: int = 4) -> str:
    api_key = get_api_key()
    if not api_key:
        return "[Gemini API key not configured.]"

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 1024}
    }
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=40)

            if response.status_code == 429:
                # Exponential backoff: 5s, 10s, 20s, 40s
                wait = 5 * (2 ** attempt)
                time.sleep(wait)
                continue

            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(5)
                continue
            return "[Request timed out after multiple attempts.]"
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < retries - 1:
                time.sleep(5 * (2 ** attempt))
                continue
            return f"[API error {response.status_code}: {str(e)}]"
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return f"[Failed to parse Gemini response: {str(e)}]"
        except Exception as e:
            return f"[Unexpected error: {str(e)}]"

    return "[Rate limit exceeded after retries. Wait 60 seconds and try again.]"


SYSTEM_INSTRUCTION = """You are a senior AI fairness and ethics researcher with expertise in:
- Machine learning bias detection and mitigation
- Employment and lending law (ECOA, Title VII, Fair Housing Act)
- EU AI Act compliance for high-risk AI systems
- Statistical fairness metrics and their trade-offs

Provide precise, factual, professional analysis. Do not use emojis or emdashes.
Cite specific laws and metrics where relevant. Be concise but thorough.
Never hallucinate legal citations -- only reference laws you are certain exist."""


def generate_triple_justification(
    feature_name: str,
    mi_score: float,
    proxy_target: str,
    domain: str,
    dataset_context: str
) -> dict:
    prompt = f"""Analyze this machine learning feature for fairness risks:

Feature: {feature_name}
Mutual Information Score with outcome: {mi_score:.4f}
Potential proxy for: {proxy_target}
Application domain: {domain}
Dataset context: {dataset_context}

Generate a triple justification in exactly this JSON format (no markdown, raw JSON only):
{{
  "statistical": "2-3 sentences on the statistical relationship, MI interpretation, and variance explained",
  "moral_historical": "2-3 sentences on historical discrimination patterns, documented societal harms, and ethical concerns with using this feature",
  "legal": "2-3 sentences citing specific applicable laws (ECOA, Title VII, EU AI Act Article numbers, etc.) and how this feature may create disparate impact or violate protected class rules",
  "risk_level": "HIGH or MEDIUM or LOW",
  "recommended_action": "One concrete recommended action"
}}"""

    result = call_gemini(prompt, SYSTEM_INSTRUCTION, temperature=0.2)

    try:
        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "statistical": result[:300] if len(result) > 300 else result,
            "moral_historical": "Could not parse structured response.",
            "legal": "Could not parse structured response.",
            "risk_level": "UNKNOWN",
            "recommended_action": "Manual review recommended."
        }


def generate_fairness_rationale(
    chosen_metric: str,
    metric_values: dict,
    domain: str,
    sensitive_attribute: str,
    accuracy_cost: float
) -> str:
    metric_str = "\n".join([f"  - {k}: {v:.4f}" for k, v in metric_values.items()])
    prompt = f"""A compliance officer has selected '{chosen_metric}' as the fairness criterion for a {domain} automated decision system.

Sensitive attribute: {sensitive_attribute}
Fairness metrics computed:
{metric_str}
Accuracy cost: {accuracy_cost:.2f}%

Write a professional audit rationale paragraph (150-200 words) for regulatory documentation.
Explain why this fairness definition was selected, trade-offs accepted, and regulatory alignment.
Formal third-person prose. No bullets, no emojis, no emdashes."""

    return call_gemini(prompt, SYSTEM_INSTRUCTION, temperature=0.3)


def generate_blackbox_interpretation(
    feature_changes: list,
    original_outcome: str,
    new_outcome: str,
    domain: str
) -> str:
    changes_str = "\n".join(
        [f"  - {c['feature']}: {c['original']} -> {c['changed']}" for c in feature_changes]
    )
    prompt = f"""In a {domain} automated decision system, a counterfactual analysis was performed.

Original decision: {original_outcome}
Changes made:
{changes_str}
New decision: {new_outcome}

Write a 2-3 sentence professional interpretation of what this reveals about potential bias.
No emojis or emdashes."""

    return call_gemini(prompt, SYSTEM_INSTRUCTION, temperature=0.3)


def generate_audit_summary(
    domain: str,
    sensitive_attributes: list,
    top_proxies: list,
    chosen_fairness_metric: str,
    key_findings: list
) -> str:
    findings_str = "\n".join([f"  - {f}" for f in key_findings])
    prompt = f"""Generate an executive summary for a fairness audit report.

Domain: {domain}
Sensitive attributes: {', '.join(sensitive_attributes)}
Top proxy risks: {', '.join(top_proxies)}
Fairness criterion: {chosen_fairness_metric}
Key findings:
{findings_str}

Write a 200-250 word executive summary for a board-level compliance report.
Three paragraphs: methodology, findings, recommendations.
Formal language. No emojis or emdashes."""

    return call_gemini(prompt, SYSTEM_INSTRUCTION, temperature=0.3)
