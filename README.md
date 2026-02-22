# Simon (macOS, Python)

Native‑like macOS desktop app in Python with speech in/out and local/API LLM.

## Features (MVP)
- Push‑to‑talk STT (Deutsch/Englisch) via faster‑whisper (local) with model/device/preset selection
- TTS via macOS `say` (local), optional Piper HTTP
- Chat UI with history, auto‑send, auto‑speak
- LLM: local Ollama or OpenAI‑compatible API
- Mic monitor + Auto‑Test

## Install
Recommended: Python 3.12 (best compatibility for audio + faster‑whisper).

If you only have Python 3.14, install Python 3.12 via Homebrew:
```bash
brew install python@3.12
```

Create venv with Python 3.12:
```bash
brew install portaudio
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python simon.py
```

## App Bundle (Double‑Click)
A runnable app bundle is created at:
`/Users/holger/simon-python/Simon.app`

If macOS blocks it on first launch:
- Right‑click the app → Open
- Confirm "Open"

## Build Native .app (embedded Python, no Terminal)
```bash
./build_app.sh
```
This produces: `Simon.app`

The build script generates a green **S‑monogram icon** automatically.

## Auto Updates (Optional)
Default update URL (GitHub Pages):
`https://HolliOnRoad.github.io/Simon/updates/simon.json`

Set an update URL that returns JSON like:
```
{"version":"1.0.1","url":"https://example.com/Simon.app.zip"}
```
Then enable **Auto‑Check Updates** or click **Check Now**.

### GitHub Release Flow
1. Build the app: `./build_app.sh`
2. Create release assets + update JSON: `./release.sh`
3. Upload `dist/Simon.app.zip` to your GitHub Release
4. Commit `updates/simon.json` to your repo (or host it via GitHub Pages)

Set these environment variables before running `release.sh`:
```bash
export GITHUB_REPO="HolliOnRoad/Simon"
export TAG="v1.0.0"
```

## Codesign + Notarize (Recommended)
```bash
export DEVELOPER_ID_APP="Developer ID Application: Your Name (TEAMID)"
export NOTARY_PROFILE="simon-notary"
./sign_and_notarize.sh
```

To create a keychain profile:
```bash
xcrun notarytool store-credentials "simon-notary" \
  --apple-id "you@example.com" \
  --team-id "TEAMID" \
  --password "app-specific-password"
```

## Troubleshooting
- If microphone doesn’t work: System Settings → Privacy & Security → Microphone → enable for Python.
- If STT is slow: change model name in the UI (e.g. `base` or `tiny`).
- Select the correct input device under `Input Device` if the mic isn't working.
- Use `Mic Monitor` to see live input level, and `Auto‑Test` to verify the device quickly.

## GitHub Setup
```bash
git init

git add .

git commit -m "Initial commit"

git branch -M main

git remote add origin https://github.com/HolliOnRoad/Simon.git

git push -u origin main
```

Repo: `HolliOnRoad/Simon`.

## GitHub Pages (for updates)
Enable GitHub Pages:

1. Repo → **Settings** → **Pages**
2. **Source**: Deploy from a branch
3. **Branch**: `main` and **/docs**

Your update URL will be:

`https://HolliOnRoad.github.io/Simon/updates/simon.json`

## GitHub Actions (Release Build)
Workflow file: `.github/workflows/release.yml`

Create a tag to trigger a build + release upload:
```bash
git tag v1.0.0
git push origin v1.0.0
```

This will:
- build the app on macOS
- upload `Simon.app.zip` to the GitHub Release
- update `updates/simon.json` on `main`
