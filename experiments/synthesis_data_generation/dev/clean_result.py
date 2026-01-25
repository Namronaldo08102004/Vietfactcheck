import json
import os
import sys

# ------------------------------------------------------------------
# Path handling
# ------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

def map_fake_context_by_line_order(
    full_context_path: str,
    fake_context_txt_path: str,
    output_path: str
):
    # --------------------------------------------------
    # Load full context (statement + topic)
    # --------------------------------------------------
    with open(full_context_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    # --------------------------------------------------
    # Load fake contexts (one per line)
    # --------------------------------------------------
    with open(fake_context_txt_path, "r", encoding="utf-8") as f:
        fake_contexts = json.load(f)


    assert len(full_data) == len(fake_contexts), (
        f"❌ Mismatch: {len(full_data)} records vs {len(fake_contexts)} fake contexts"
    )

    # --------------------------------------------------
    # Merge by order
    # --------------------------------------------------
    merged = []

    for i, (record, fake_ctx) in enumerate(zip(full_data, fake_contexts)):
        merged.append({
            "index": record.get("index", i),
            "topic": record.get("topic"),
            "statements": record.get("statements"),
            "fake_context": fake_ctx
        })

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"✅ Mapped {len(merged)} fake contexts")
    print(f"📁 Saved to: {output_path}")

map_fake_context_by_line_order(
    full_context_path= os.path.join(PROJECT_ROOT, "experiments/synthesis_data_generation/dev/results/dev_parse_data/dev_grouped_full_context.json"),
    fake_context_txt_path= os.path.join(PROJECT_ROOT, "experiments/synthesis_data_generation/dev/results/dev_synthesis_data/dev_fake_context.json"),
    output_path=os.path.join(PROJECT_ROOT, "experiments/synthesis_data_generation/dev/results/dev_cleaned/dev_cleaned.json"),
)
