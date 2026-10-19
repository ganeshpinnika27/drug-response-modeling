#!/usr/bin/env bash
# Download public inputs. Run from the repo root.
#
# GDSC2 (Sanger, release 8.5) is a direct link.
# DepMap 24Q4 Public (doi:10.25452/figshare.plus.27993248) and PRISM
# Repurposing 20Q2 secondary screen (figshare article 20564034) come from
# figshare, pinned by file ID and checked by MD5, because the DepMap portal's
# download links sit behind a browser check.
set -euo pipefail
mkdir -p data/raw
cd data/raw

GDSC_URL="https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx"
if [[ ! -s GDSC2_fitted_dose_response.xlsx ]]; then
  echo "[gdsc] downloading GDSC2 fitted dose response"
  curl -fL "$GDSC_URL" -o GDSC2_fitted_dose_response.xlsx
fi

md5_of() { if command -v md5sum >/dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi; }

# name  figshare_file_id  md5
while read -r name fid md5; do
  if [[ -s "$name" && "$(md5_of "$name")" == "$md5" ]]; then
    echo "[figshare] ok      $name"
    continue
  fi
  echo "[figshare] downloading $name"
  curl -fL "https://ndownloader.figshare.com/files/$fid" -o "$name.part"
  got=$(md5_of "$name.part")
  if [[ "$got" != "$md5" ]]; then
    echo "[figshare] MD5 mismatch for $name (expected $md5, got $got)" >&2
    exit 1
  fi
  mv "$name.part" "$name"
  echo "[figshare] ok      $name"
done <<'EOF'
Model.csv                                      51065297 675210d17675f3517b0ce39a3c274f16
OmicsExpressionProteinCodingGenesTPMLogp1.csv  51065489 71794802b750ce77c422dad0720a40af
OmicsSomaticMutationsMatrixHotspot.csv         51065750 a5aeb1deef897ead3c955c372148d840
prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv  36794595 eb0c7ee3ccb9e480148c71fcdc97312e
prism-repurposing-20q2-secondary-screen-cell-line-info.csv                  36794589 4c3a5fbee769e8384064a8fbe4b4ed7d
prism-repurposing-20q2-secondary-screen-readme                              36794613 47419770d1edfcc36e0b11465f7d1fd0
EOF
