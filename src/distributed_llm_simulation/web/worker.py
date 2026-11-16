"""One isolated Web job; deliberately uses the same application service as CLI."""

import json
import sys
from pathlib import Path

from ..application import run_experiment
from .drafts import Draft, parse_draft


def main():
    draft_path, output = map(Path, sys.argv[1:3])
    draft = Draft.model_validate_json(draft_path.read_text())
    result = run_experiment(*parse_draft(draft), output)
    print(json.dumps({"status": result.status, "makespan_s": result.makespan_s}), flush=True)


if __name__ == "__main__":
    main()
