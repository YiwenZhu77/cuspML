#!/bin/bash
# Dual-model critical review of the revised JGR manuscript + response letter.
# Model A = GPT-5.4 via codex CLI (strong critical reviewer).
# Model B = DeepSeek via API (independent second opinion).
# Each is prompted as a skeptical JGR-Space Physics reviewer AFTER the revision, told
# the reviewer comments have supposedly been addressed, and asked to find what is still
# wrong: unsupported claims, internal number inconsistencies across text/tables/figures,
# methodological weakness, and any response that doesn't actually answer the reviewer.
set -u
P=/glade/work/yizhu/cuspML/paper
OUT=/glade/work/yizhu/cuspML/output/review
mkdir -p "$OUT"
export PATH=~/.TinyTeX/bin/x86_64-linux:$PATH

pdftotext -layout "$P/main.pdf" "$OUT/manuscript.txt" 2>/dev/null
pdftotext -layout "$P/response_to_reviewers.pdf" "$OUT/response.txt" 2>/dev/null

PROMPT=$(cat <<'EOP'
You are a rigorous, skeptical reviewer for JGR: Space Physics. This manuscript
("Predicting Ionospheric Cusp Location from Solar Wind: An XGBoost Model...") was
returned for MAJOR revision. Below are (1) the REVISED manuscript and (2) the authors'
RESPONSE letter claiming every comment is addressed. Your job is to find what is STILL
wrong or weak. Be specific and adversarial. Focus on:
  - Internal number inconsistencies: any value that differs between the abstract, text,
    tables, figures, or the response letter (e.g. two different MAE for the same setting).
  - Claims not supported by the presented evidence, or overclaiming.
  - Methodological weaknesses a reviewer would still object to (splits, baselines,
    controls, metric choices, sample sizes).
  - Responses that dodge, only partially answer, or contradict the manuscript text.
  - Logic/consistency of the conclusions.
Do NOT praise. Do NOT restate the paper. Output ONLY a numbered list. For each item:
  [SEVERITY: blocking | major | minor] one-sentence issue — where it appears — the fix.
List each DISTINCT issue exactly ONCE; never repeat an issue. If nothing is blocking or major, write "NO BLOCKING/MAJOR ISSUES" then list minors.
EOP
)

BODY="$PROMPT

===== REVISED MANUSCRIPT =====
$(cat "$OUT/manuscript.txt")

===== RESPONSE LETTER =====
$(cat "$OUT/response.txt")
"

echo "$BODY" > "$OUT/review_input.txt"
echo "input chars: $(wc -c < "$OUT/review_input.txt")"

# ---- Model A: GPT-5.4 via codex ----
echo "[A] codex GPT-5.4 reviewing..."
codex exec --skip-git-repo-check -s read-only - < "$OUT/review_input.txt" > "$OUT/review_gpt.md" 2>"$OUT/review_gpt.err"
echo "[A] done -> $OUT/review_gpt.md ($(wc -l < "$OUT/review_gpt.md") lines)"

# ---- Model B: DeepSeek ----
echo "[B] DeepSeek reviewing..."
python3 - "$OUT/review_input.txt" "$OUT/review_ds.md" <<'PY'
import sys, json, urllib.request, os
inp, outp = sys.argv[1], sys.argv[2]
body = open(inp).read()
req = urllib.request.Request(
    "https://api.deepseek.com/chat/completions",
    data=json.dumps({"model":"deepseek-chat",
        "messages":[{"role":"user","content":body}],
        "max_tokens":1800,"temperature":0.1}).encode(),
    headers={"Content-Type":"application/json",
             "Authorization":"Bearer "+os.environ["DEEPSEEK_API_KEY"]})
try:
    r = json.load(urllib.request.urlopen(req, timeout=180))
    open(outp,"w").write(r["choices"][0]["message"]["content"])
    print("[B] deepseek ok")
except Exception as e:
    open(outp,"w").write(f"DEEPSEEK ERROR: {e}")
    print("[B] deepseek error:", e)
PY
echo "[B] done -> $OUT/review_ds.md"
echo "=== DUAL REVIEW DONE ==="
