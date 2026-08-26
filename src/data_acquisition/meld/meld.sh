#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$SCRIPT_DIR"
ARCHIVE="$DEST_DIR/MELD.Raw.tar.gz"
URL="https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz"

# download if missing
if [ ! -f "$ARCHIVE" ]; then
  echo "Downloading $URL to $ARCHIVE"
  wget --no-check-certificate -O "$ARCHIVE" "$URL"
else
  echo "Archive already exists at $ARCHIVE"
fi

# extract top-level archive into DEST_DIR
echo "Extracting $ARCHIVE to $DEST_DIR"
tar -xzvf "$ARCHIVE" -C "$DEST_DIR"

# find and extract any inner .tar.gz files (train/dev/test)
shopt -s nullglob
for inner in "$DEST_DIR"/*.tar.gz "$DEST_DIR"/*/*.tar.gz; do
  [ -f "$inner" ] || continue
  tar -xzf "$inner" -C "$DEST_DIR"
done
shopt -u nullglob