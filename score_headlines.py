"""CLI tool: score news headlines (Optimistic / Pessimistic / Neutral)."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import List

import joblib
from sentence_transformers import SentenceTransformer


# friendly hint (for this assignment)
def usage_exit() -> None:
    """Print CLI usage information and exit with error code 1."""
    print(
        "\nUsage:  python score_headlines.py <headlines_file> <source_tag>\n"
        "  headlines_file : text file with **one headline per line**\n"
        "  source_tag     : identifier used in output filename "
        "(e.g. nyt, chicagotribune)\n\n"
        "Example:\n"
        "  python score_headlines.py todaysheadlines.txt nyt\n"
    )
    sys.exit(1)


def load_headlines(path: Path) -> List[str]:
    """Load headlines from a text file, one headline per line."""
    if not path.exists():
        print(f"Error: input file '{path}' not found.")
        sys.exit(1)
    with path.open(encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    if not lines:
        print("Error: the input file is empty.")
        sys.exit(1)
    return lines


def get_sentence_embedder() -> SentenceTransformer:
    """Return a MiniLM sentence-transformer (local path if present, else download)."""
    local = Path("/opt/huggingface_models/all-MiniLM-L6-v2")
    if local.exists():
        return SentenceTransformer(str(local))
    # auto download if does not have it
    return SentenceTransformer("all-MiniLM-L6-v2")


def main() -> None:
    """Command-line entry: parse args, embed headlines, predict, write output file."""
    if len(sys.argv) != 3:
        print("Error: incorrect number of arguments.\n")
        usage_exit()

    infile = Path(sys.argv[1])
    source = sys.argv[2]

    headlines = load_headlines(infile)

    embedder = get_sentence_embedder()
    embeddings = embedder.encode(headlines, convert_to_numpy=True)

    svm_path = Path(__file__).with_name("svm.joblib")
    if not svm_path.exists():
        print("Error: 'svm.joblib' not found in the same directory as this script.")
        sys.exit(1)
    svm = joblib.load(svm_path)

    predictions = svm.predict(embeddings)

    out_name = f"headline_scores_{source}_{date.today():%Y_%m_%d}.txt"
    with open(out_name, "w", encoding="utf-8") as fh:
        for label, headline in zip(predictions, headlines, strict=True):
            fh.write(f"{label},{headline}\n")

    print(f"Success!  Results written to '{out_name}'.")


if __name__ == "__main__":
    main()
