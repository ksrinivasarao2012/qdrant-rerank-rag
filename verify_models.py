"""
Pre-flight check for the active prompt variants.

Run this after any model change, and before Groq's next deprecation date:

    .\.venv\Scripts\python.exe verify_models.py

It confirms:
  1. The active answer + vision model IDs still exist on Groq.
  2. Whether gpt-oss reasoning tokens leak into `content` (which would show up
     in the chat UI and break the "do not have enough information" check in
     app.py's citation logic).
  3. That retired variants are in fact retired, not silently still in use.
"""
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

from backend.core import prompts  # noqa: E402

ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
KEY = (os.getenv("GROQ_API_KEY") or "").strip()


def call(model, messages, extra=None, max_tokens=150):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if extra:
        body.update(extra)
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0"
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return True, json.load(r)
    except urllib.error.HTTPError as e:
        return False, e.read().decode()[:400]
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    if not KEY:
        print("ERROR: GROQ_API_KEY not set in .env")
        return 1

    failures = []

    # ---------------------------------------------------------------- answer
    cfg = prompts.get("answer")
    variant = prompts.active("answer")
    print(f"=== ANSWER: {cfg['model']}  [variant={variant}] ===")
    print(f"    extra_body: {cfg.get('extra_body')}")

    context = prompts.render(
        "answer", "citation_block",
        index=1, source="sample.pdf", page=2,
        snippet="The Qdrant collection uses 384 dimensions with cosine distance.",
    )
    messages = [
        {"role": "system", "content": cfg["system"]},
        {"role": "user", "content": prompts.render(
            "answer", "user", context=context,
            query="How many dimensions does the collection use?")},
    ]

    ok, res = call(cfg["model"], messages, cfg.get("extra_body"))
    if not ok:
        failures.append(f"answer model {cfg['model']}")
        print(f"    FAIL: {res}\n")
    else:
        msg = res["choices"][0]["message"]
        content = msg.get("content") or ""
        print("    OK: model is live")
        print(f"    answer: {content[:160]!r}")

        # Does reasoning leak into the visible stream?
        leaked = [k for k in ("reasoning", "reasoning_content") if msg.get(k)]
        if leaked:
            print(f"    NOTE: reasoning returned in separate field(s) {leaked} "
                  f"-- these are NOT in `content`, so the chat UI is safe.")
        if "<think" in content.lower() or "analysis" in content[:40].lower():
            failures.append("reasoning tokens leaking into content")
            print("    WARN: reasoning may be leaking into `content` -- inspect above.")
        else:
            print("    OK: `content` looks clean (no visible reasoning)")

        # The grounding refusal string must survive, app.py depends on it.
        ok2, res2 = call(cfg["model"], [
            {"role": "system", "content": cfg["system"]},
            {"role": "user", "content": prompts.render(
                "answer", "user", context=context,
                query="What is the CEO's home address?")},
        ], cfg.get("extra_body"))
        if ok2:
            refusal = (res2["choices"][0]["message"].get("content") or "").lower()
            if "do not have enough information" in refusal:
                print("    OK: refusal string intact (app.py citation check works)")
            else:
                failures.append("refusal string changed")
                print(f"    WARN: expected refusal phrasing not found -> {refusal[:120]!r}")
        print(f"    tokens: {res.get('usage', {}).get('total_tokens')}\n")

    # ---------------------------------------------------------------- vision
    vc = prompts.get("vision_summary")
    print(f"=== VISION: {vc['model']}  [variant={prompts.active('vision_summary')}] ===")
    ok, res = call(vc["model"], [{"role": "user", "content": "Reply with one word: ok"}])
    if ok:
        print("    OK: model is live\n")
    else:
        failures.append(f"vision model {vc['model']}")
        print(f"    FAIL: {res}\n")

    # ------------------------------------------------------- retired models
    print("=== RETIRED (expected to fail after their shutdown dates) ===")
    active_models = {cfg["model"], vc["model"]}
    for group in ("answer", "vision_summary"):
        for name in prompts.variants(group):
            m = prompts.get(group, name)["model"]
            if m in active_models:
                continue
            ok, _ = call(m, [{"role": "user", "content": "hi"}], max_tokens=5)
            print(f"    {m:<34} {'still live' if ok else 'dead (expected)'}")

    print()
    if failures:
        print(f"FAILURES: {failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
