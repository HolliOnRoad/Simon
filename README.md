# Simon (macOS, Python)

Native macOS-Desktop-App in Python mit Sprachein- und -ausgabe sowie lokaler/API-LLM-Anbindung.

## Funktionen (MVP)
- Push-to-talk STT (Deutsch/Englisch) via faster-whisper (lokal) mit Modell/Geraet/Preset-Auswahl
- TTS via macOS `say` (lokal), optional Piper HTTP
- Chat-UI mit Verlauf, Auto-Send, Auto-Speak
- LLM: lokal (Ollama) oder OpenAI-kompatible API
- Mic-Monitor + Auto-Test

## Installation
Empfohlen: Python 3.12 (beste Kompatibilitaet fuer Audio + faster-whisper).

Falls du nur Python 3.14 hast, installiere Python 3.12 via Homebrew:
```bash
brew install python@3.12
```

Virtuelle Umgebung mit Python 3.12:
```bash
brew install portaudio
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Start
```bash
python simon.py
```

## App-Bundle (Doppelklick)
Ein startbares App-Bundle liegt hier:
`/Users/holger/simon-python/Simon.app`

Wenn macOS blockt:
- Rechtsklick → Oeffnen
- "Oeffnen" bestaetigen

## Native .app bauen (embedded Python, ohne Terminal)
```bash
./build_app.sh
```
Ergebnis: `Simon.app`

Das Build-Skript erzeugt automatisch ein gruens S-Monogramm-Icon.

## Auto-Updates (optional)
Standard-Update-URL (GitHub Pages):
`https://HolliOnRoad.github.io/Simon/updates/simon.json`

Update-JSON-Format:
```
{"version":"1.0.1","url":"https://example.com/Simon.app.zip"}
```
Dann in der App **Updates automatisch pruefen** aktivieren oder im Menue **Simon → Nach Updates suchen...**.

### GitHub Release Flow
1. App bauen: `./build_app.sh`
2. Release-Artefakte + Update-JSON: `./release.sh`
3. `dist/Simon.app.zip` und `dist/Simon.dmg` im GitHub Release hochladen
4. `updates/simon.json` committen (oder via GitHub Pages ausliefern)

Umgebungsvariablen fuer `release.sh`:
```bash
export GITHUB_REPO="HolliOnRoad/Simon"
export TAG="v1.0.0"
```

## GitHub Pages (Updates)
Aktivieren:
1. Repo → **Settings** → **Pages**
2. **Source**: Deploy from a branch
3. **Branch**: `main` und **/docs**

Update-URL:
`https://HolliOnRoad.github.io/Simon/updates/simon.json`

## GitHub Actions (Release Build)
Workflow: `.github/workflows/release.yml`

Tag setzen, um Build + Release zu starten:
```bash
git tag v1.0.0
git push origin v1.0.0
```

Das erzeugt:
- `Simon.app.zip` (fuer Auto-Update)
- `Simon.dmg` (fuer manuellen Download)
- aktualisiert `updates/simon.json`

## Codesign + Notarisierung (optional, kostet Apple-Account)
```bash
export DEVELOPER_ID_APP="Developer ID Application: Your Name (TEAMID)"
export NOTARY_PROFILE="simon-notary"
./sign_and_notarize.sh
```

Keychain-Profil anlegen:
```bash
xcrun notarytool store-credentials "simon-notary" \
  --apple-id "you@example.com" \
  --team-id "TEAMID" \
  --password "app-specific-password"
```

## Troubleshooting
- Mikrofon geht nicht: System Settings → Privacy & Security → Microphone → Python erlauben
- STT zu langsam: Modell in der UI auf `base` oder `tiny` stellen
- Richtiges Eingabegeraet unter `Input Device` waehlen
- `Mic Monitor` zeigt Live-Pegel, `Auto-Test` prueft das Geraet

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
