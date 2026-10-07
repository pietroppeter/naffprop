#!/bin/sh
# Builds the explanation pages into docs/explain/site, as CI does (run from
# anywhere, with nim on PATH, e.g. from `uv pip install nimlang`):
#   sh docs/explain/build.sh
# nimib and karax come from git at pinned commits, into $NIMIB_DEPS (default
# docs/explain/deps, not committed); config.nims puts them on the path. Linux
# needs libpcre3 (nim-markdown uses std/re).
set -e
here=$(cd "$(dirname "$0")" && pwd)
export NIMIB_DEPS=${NIMIB_DEPS:-$here/deps}
mkdir -p "$NIMIB_DEPS"
while read -r name repo commit; do
  d="$NIMIB_DEPS/$name"
  if [ "$(git -C "$d" rev-parse HEAD 2>/dev/null)" != "$commit" ]; then
    rm -rf "$d" && git init -q "$d"
    git -C "$d" fetch -q --depth 1 "https://github.com/$repo" "$commit"
    git -C "$d" checkout -q FETCH_HEAD
  fi
done <<DEPS
nimib pietroppeter/nimib 46ebe3477057e2d07604391e6dea602debb058dc
karax karaxnim/karax fa5fdef42b3dd290880b3e4892e52385b892cfcb
fusion nim-lang/fusion 562467452b32cb7a97410ea177f083e6d8405734
nim-markdown soasme/nim-markdown 995405b78fb49b7658b7f0d94efe1b52f86c0a85
parsetoml NimParsers/parsetoml d297f5a81be2472905d367aa675dcbbca6e7a5d2
jsony treeform/jsony c5874fb71435e3ced8ee0025822e15f5d0e8e622
DEPS
bin=$(mktemp -d)
nim c -r --hints:off -o:"$bin/gen_data" "$here/gen_data.nim"
nim c -r --hints:off -o:"$bin/test_explain" "$here/test_explain.nim"
nim js -d:nodejs -r --hints:off -o:"$bin/test_explain.js" "$here/test_explain.nim"
# nimib writes the page next to the cwd: build from here
(cd "$here" && nim c -r --hints:off -o:"$bin/fig1a" fig1a.nim)
mkdir -p "$here/site"
cp "$here/fig1a.html" "$here/site/fig1a.html"
cp "$here/fig1a.html" "$here/site/index.html"
rm -rf "$bin"
