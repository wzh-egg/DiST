#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "$0")" && pwd)"
usage() {
    echo "bash run.sh preprocess COORDINATES.npy CANDIDATES.npz GEOMETRY.npz"
    echo "bash run.sh train|test STATION_DIR GRID_DIR GEOMETRY.npz RUN_DIR"
}
case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    preprocess)
        [[ $# == 4 ]] || { usage >&2; exit 2; }
        exec "${PYTHON:-python}" "$root/preprocess.py" "$2" "$3" "$4" ;;
    train|test)
        case "$#" in
            5) inputs=(--stations "$2" --grid "$3" --geometry "$4" --out "$5") ;;
            4) inputs=(--data "$2" --geometry "$3" --out "$4") ;;
            *) usage >&2; exit 2 ;;
        esac
        entry=train.py
        if [[ $1 == test ]]; then entry=evaluate.py; fi
        exec "${PYTHON:-python}" -Bu "$root/$entry" "${inputs[@]}" ;;
    *) usage >&2; exit 2 ;;
esac
