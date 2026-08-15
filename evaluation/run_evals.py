import subprocess
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration runs to execute
RUNS = [
    {
        "name": "1. Baseline (Hybrid + Rerank)",
        "args": ["--local"]
    },
    {
        "name": "2. No Reranker (Hybrid Only)",
        "args": ["--local", "--no_rerank"]
    },
    {
        "name": "3. Sparse Only (BM25 + Rerank)",
        "args": ["--local", "--method", "sparse"]
    },
    {
        "name": "4. Fully Optimized (Hybrid + Rerank + Rewrite)",
        "args": ["--local", "--rewrite"]
    }
]

def run_benchmarks():
    print("=== STARTING ABLATION SUITE BENCHMARKS ===")
    results_summary = []

    for run in RUNS:
        print(f"\n>>> Running Config: {run['name']}...")
        cmd = [sys.executable, "evaluation/eval_retriever.py"] + run["args"]
        
        try:
            # We use Popen to read the output and output character by character
            # to handle carriage returns (\r) correctly for progress bars.
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=0
            )
            
            output_lines = []
            current_line = []
            
            while True:
                char = process.stdout.read(1)
                if not char and process.poll() is not None:
                    break
                if char:
                    sys.stdout.write(char)
                    sys.stdout.flush()
                    if char == '\n':
                        output_lines.append("".join(current_line))
                        current_line = []
                    elif char != '\r':
                        current_line.append(char)
            
            if current_line:
                output_lines.append("".join(current_line))
            
            process.communicate() # wait for process finish
            
            # Parse the metrics from the captured output lines
            mrr, r3, r5, r10 = None, None, None, None
            for line in output_lines:
                if "MRR=" in line:
                    parts = line.strip().split()
                    for p in parts:
                        if p.startswith("MRR="):
                            mrr = p.split("=")[1]
                        elif p.startswith("recall@3="):
                            r3 = p.split("=")[1]
                        elif p.startswith("recall@5="):
                            r5 = p.split("=")[1]
                        elif p.startswith("recall@10="):
                            r10 = p.split("=")[1]
            
            results_summary.append({
                "config": run["name"],
                "mrr": mrr or "N/A",
                "r3": r3 or "N/A",
                "r5": r5 or "N/A",
                "r10": r10 or "N/A"
            })
        except Exception as e:
            print(f"Error running configuration {run['name']}: {e}")
            results_summary.append({
                "config": run["name"],
                "mrr": "FAILED",
                "r3": "FAILED",
                "r5": "FAILED",
                "r10": "FAILED"
            })

    # Output the final ablation comparison table
    print("\n" + "="*70)
    print("                      RETRIEVAL ABLATION COMPARISON TABLE")
    print("="*70)
    print(f"{'Configuration':<45} | {'MRR':<6} | {'Recall@3':<8} | {'Recall@5':<8} | {'Recall@10':<8}")
    print("-"*70)
    for row in results_summary:
        print(f"{row['config']:<45} | {row['mrr']:<6} | {row['r3']:<8} | {row['r5']:<8} | {row['r10']:<8}")
    print("="*70)

if __name__ == "__main__":
    run_benchmarks()
