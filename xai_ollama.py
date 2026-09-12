import os
import json
import base64
import requests
from datetime import datetime
from typing import Optional, Dict, Any

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3:4b"

def _b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _safe_json_extract(text: str) -> Dict[str, Any]:
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError("JSON not found in LLM output.")
    return json.loads(text[s:e+1])

def build_xai_prompt(img_name, pred_count, gt_count, evidence):
    gt_txt = "null" if gt_count is None else int(gt_count)
    return f"""
You are an expert in explainable AI for dense crowd counting (CSRNet).
Return ONLY valid JSON. All keys must be present.

Context:
- Image: {img_name}
- Predicted count: {int(pred_count)}
- Ground truth: {gt_txt}

Numerical evidence:
{json.dumps(evidence, ensure_ascii=False)}

Schema:
{{
  "summary": "string",
  "visual_evidence": ["string", "string"],
  "error_analysis": ["string", "string"],
  "actionable_improvements": ["string", "string"],
  "confidence_and_caveats": ["string", "string"]
}}
""".strip()

def generate_xai_report(
    img_name: str,
    pred_count: int,
    gt_count: Optional[int],
    evidence: Dict[str, Any],
    out_md_path: str,
    model: str = DEFAULT_MODEL,
):
    prompt = build_xai_prompt(img_name, pred_count, gt_count, evidence)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json"
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    text = r.json()["message"]["content"]
    data = _safe_json_extract(text)

    md = f"# XAI Report — {img_name}\n\n"
    md += f"- Model: {model}\n"
    md += f"- Predicted: {pred_count}\n"
    md += f"- GT: {'N/A' if gt_count is None else gt_count}\n"
    md += f"- Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"

    md += "## Evidence Summary (Auto)\n"
    for k, v in evidence.items():
        md += f"- {k}: {v}\n"
    md += "\n"

    sections = ["summary", "visual_evidence", "error_analysis", 
                "actionable_improvements", "confidence_and_caveats"]

    for sec in sections:
        md += f"## {sec.replace('_',' ').title()}\n"
        content = data.get(sec, "Information not provided by model.")
        if isinstance(content, list):
            for x in content:
                md += f"- {x}\n"
        else:
            md += str(content) + "\n"
        md += "\n"

    os.makedirs(os.path.dirname(out_md_path), exist_ok=True)
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(md)
    return out_md_path

def start_visual_chat(
    original_img: str,
    heatmap_img: str,
    overlay_img: str,
    numeric_context: dict,
    xai_text: str,
    model: str = DEFAULT_MODEL,
):
    images = [
        _b64_image(original_img),
        _b64_image(heatmap_img),
        _b64_image(overlay_img),
    ]

    system_msg = (
        "You are a multimodal explainability assistant for density crowd counting.\n"
        "Base all answers on the provided numeric facts, heatmap and overlay images.\n"
    )

    facts_msg = (
        "=== FIXED FACTS ===\n"
        f"Predicted count: {numeric_context['predicted']}\n"
        f"Ground truth count: {numeric_context['gt']}\n"
        f"Absolute error: {numeric_context['abs_error']}\n\n"
        "=== Evidence ===\n"
        f"{json.dumps(numeric_context['evidence'])}\n\n"
        "=== XAI Report ===\n"
        f"{xai_text}\n"
    )

    print("\n=== Multimodal XAI Chat (type 'exit' to quit) ===")
    history = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": facts_msg, "images": images}
    ]

    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        history.append({"role": "user", "content": q})
        payload = {
            "model": model,
            "messages": history,
            "stream": False
        }

        r = requests.post(OLLAMA_URL, json=payload, timeout=180)
        r.raise_for_status()
        ans = r.json()["message"]["content"]
        print("\nAssistant:", ans)
        history.append({"role": "assistant", "content": ans})