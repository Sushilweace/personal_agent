# Personal AI Agent

A modular, fully local personal assistant built with **LangGraph + LangChain**, powered by
**Ollama** (`llama3.2:3b`) and **ChromaDB** (`nomic-embed-text` embeddings). Sized for a machine
with ~24 GB RAM and a 4 GB GPU.

```
          ┌─────────┐  tool_calls?   ┌─────────┐
 START ──▶│  agent  │───── yes ─────▶│  tools  │
          │ (LLM)   │◀───────────────│ (ToolNode)
          └────┬────┘                └─────────┘
               │ no
               ▼
              END            state is checkpointed per thread_id (MemorySaver)
```

## Project layout

```
personal_agent/
├── data/documents/      your .txt / .md / .pdf files (a sample note is included - delete it)
│   └── job_profile.md   your job requirements (roles, skills, locations, avoid list)
├── memory/              persistent ChromaDB storage
├── tools/
│   ├── system_tools.py  get_current_time, run_calculator, get_system_info
│   ├── notes_tools.py   query_personal_notes  (semantic search via rag.py)
│   ├── jobs_tools.py    read_job_alerts (Gmail job emails ranked against your profile)
│   └── web_tools.py     web_search (DuckDuckGo), fetch_webpage, open_in_browser
├── config.py            Pydantic settings (env / .env overridable)
├── rag.py               load -> chunk -> embed -> Chroma; query_knowledge_base()
├── graph.py             AgentState, agent/tools nodes, routing, graph build
├── main.py              interactive rich CLI
├── web.py               browser server: live voice page (/live) + Gradio text chat (/)
├── live.py              live voice backend: speech-to-text + streamed replies
├── static/live.html     live voice page (auto speech detection, spoken replies)
├── voice.py             local speech-to-text + wake word ("Jadu")
├── jobs.py              job profile parser, job extraction from emails, ranking
├── gmail_access.py      read-only Gmail access (fixed job-sender query)
├── launcher.py          desktop-app helpers: start Ollama, open the app window
├── start_jadu.bat       double-click launcher (create_shortcut.ps1 makes Desktop/Start-menu shortcuts)
├── gmail_auth.py        one-time Google sign-in from the terminal (python gmail_auth.py)
├── connections.py       /connections page API (connect Gmail, edit requirements, run job check)
├── static/connections.html  the Connections page
├── secrets/             Google sign-in files (git-ignored, never share)
├── tests/               offline tests: python -m unittest discover -s tests
└── requirements.txt
```

## Setup

### 1. Install and start Ollama

Install from <https://ollama.com/download>. The server normally starts automatically; otherwise run
`ollama serve`.

### 2. Pull the models

```bash
ollama pull llama3.2:3b          # ~2.0 GB, chat + tool calling
ollama pull nomic-embed-text     # ~274 MB, embeddings
```

### 3. Create a virtual environment and install dependencies

Python 3.10+ (developed and tested on 3.14).

```powershell
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Add your documents

Drop `.txt`, `.md` or `.pdf` files (subfolders are fine) into `data/documents/`.

### 5. Run

```bash
python main.py
```

All commands run from the project root (the `personal_agent/` folder).

On first run, if ChromaDB is empty, your documents are indexed automatically. Startup also checks
that Ollama is reachable and both models are installed, and tells you the exact `ollama pull`
command if one is missing.

### 5b. Or run it in your browser

```powershell
python web.py
```

This serves two pages for the same agent (same tools, notes, name; one conversation per tab):

| Page | What it is |
|---|---|
| **http://127.0.0.1:7860/live** (opens automatically) | **Live voice conversation**, like Gemini Live: press *Start* once, then just talk. |
| http://127.0.0.1:7860/ | Text chat with a microphone button; tool calls appear as collapsible `🔧` bubbles. |

#### Live voice mode (`/live`)

After the single *Start* click (browsers require it to allow the microphone) there is nothing to
press: it detects when you start and stop speaking, understands you, and **speaks the answer back
sentence by sentence**, starting as soon as the first sentence is ready.

```
you speak -> auto start/stop -> local Whisper (~1-2 s) -> agent streams reply -> spoken immediately
```

- The orb shows the state: grey = off, blue = listening, green = hearing you, amber = thinking, purple = speaking.
- Any language works (auto-detected, e.g. Hindi). Replies are short and speech-friendly in this mode.
- **Settings** (remembered per browser): *Fast* (`base`) or *Accurate* (`small`, better Hindi, ~3x slower)
  recognition; *let me interrupt* (talk over it; use headphones or it hears itself); *only respond
  when I say "Jadu"*; show raw tool results; **voice engine and voice type** (see below).
- Typical latency on a modern CPU with the Indian neural voices: recognition ~1.5 s after you stop
  talking, the agent's first sentence ~1.5 s later, then ~1.5-2 s to generate its voice, so roughly
  4-5 s to the first spoken word (web searches add more). *Browser voices* skip the last step and are
  about 2 s faster. The speech and chat models are pre-loaded at startup.

##### Voices for spoken replies

| Setting | Quality | Speed | Privacy |
|---|---|---|---|
| **Indian neural voices** (default) | Natural, Indian accent | ~1.5-2 s per sentence (next sentences are prepared while one plays) | The *reply text* is sent to Microsoft's voice service (`edge-tts`); needs internet |
| **Browser voices** | Depends on your PC (Windows ships US English voices only) | Instant | Offline unless you pick an "online" voice |

- The language of each sentence is detected and matched to a voice: Hindi → Swara/Madhur
  (`hi-IN`), English → **Indian English** Neerja/Prabhat (`en-IN`), plus Tamil, Telugu, Bengali,
  Marathi, Gujarati, Kannada, Malayalam and Urdu (`*-IN`). "Voice type" switches female/male.
- Speed: `TTS_RATE` (default `+5%`, e.g. `$env:TTS_RATE="-10%"` for slower).
- If the voice service can't be reached (offline, service change), live mode falls back to browser voices
  for that sentence and tells you once.
- Your *speech recognition* stays local: your microphone audio never leaves this computer.

#### Notes

- Change the port with `WEBUI_PORT` (e.g. `$env:WEBUI_PORT=7870; python web.py`). Ctrl+C stops it.
- `VOICE_LIVE_MODEL` (default `base`) picks the live-mode model; the first use of a model downloads it.
- **Keep it local.** The agent can open pages on this computer, so the server binds to `127.0.0.1` only,
  and the voice endpoints also refuse requests without our own page's header or from another origin/host
  (so a random website in your browser can't drive the agent). Never set `WEBUI_HOST=0.0.0.0`.
- Hands-free "say Jadu any time" (`listen`) exists in the terminal; in the browser use live mode.

### 5c. Run it like a desktop app (Windows)

Double-click **`start_jadu.bat`**, or create shortcuts once:

```powershell
powershell -ExecutionPolicy Bypass -File .\create_shortcut.ps1              # Desktop + Start menu "Jadu" (custom icon)
powershell -ExecutionPolicy Bypass -File .\create_shortcut.ps1 -Minimized   # same, console starts minimized
```

Then search **Jadu** in the Start menu (right-click it to pin it to the taskbar). On launch it:

- starts Ollama in the background if it is not running (`OLLAMA_AUTOSTART=0` to disable),
- starts the server, and opens Jadu in its **own app window** (Edge/Chrome without tabs or address bar;
  `OPEN_AS_APP=0` for a normal browser tab, `AUTO_OPEN=0` to open nothing),
- does nothing twice: launching again while it runs just opens another window.

Stop it with the **⏻ Quit** button in the app (top right), Ctrl+C in the console window, or by closing
that window. Closing only the app window leaves the server (and the loaded models) running in the console.
The first microphone use asks for permission once; the app window remembers the answer.

**Phone:** Jadu runs on your PC (the models need its CPU/GPU), so the phone can't run it by itself.
Phone access (same Wi-Fi or Tailscale) is not enabled: the server listens on `127.0.0.1` only. It would
need a PIN login and HTTPS first, because the assistant can open pages on your PC and read your job emails.

## CLI commands

| Command   | Effect                                                         |
|-----------|----------------------------------------------------------------|
| `exit`    | Quit (`quit` and Ctrl+C at the prompt also work)               |
| `clear`   | Forget the conversation and start a new `thread_id`            |
| `reindex` | Wipe and rebuild the index from `data/documents/`              |
| `voice`   | Speak one command; it is transcribed locally and sent to the agent |
| `listen`  | Hands-free: only reacts to speech that includes the assistant's name (Jadu) |
| `help`    | Show the command list                                          |

Anything else is sent to the agent. Tool calls are shown live (`🔧 run_calculator(...)`).
Press Ctrl+C while the agent is working to cancel the current turn.

## Configuration

Every setting can be overridden with an environment variable or a `.env` file in the project root.

| Variable              | Default                   | Purpose                                              |
|-----------------------|---------------------------|------------------------------------------------------|
| `WEBUI_HOST` / `WEBUI_PORT` | `127.0.0.1` / `7860` | Browser UI address (`python web.py`)            |
| `ASSISTANT_NAME`      | `Jadu`                    | Assistant name and voice wake word                   |
| `WAKE_ALIASES`        | `jaadu,jadoo,jado,जादू,...` | Extra spellings of the name that wake it           |
| `OLLAMA_BASE_URL`     | `http://localhost:11434`  | Ollama server                                        |
| `LLM_MODEL`           | `llama3.2:3b`             | Chat model                                           |
| `EMBED_MODEL`         | `nomic-embed-text`        | Embedding model                                      |
| `LLM_TEMPERATURE`     | `0.1`                     | Sampling temperature                                 |
| `LLM_NUM_CTX`         | `4096`                    | Context window (raise only if VRAM allows)           |
| `LLM_NUM_PREDICT`     | `1024`                    | Max tokens per reply                                 |
| `LLM_KEEP_ALIVE`      | `600`                     | Seconds to keep the model loaded (`-1` = forever)    |
| `HISTORY_TOKEN_BUDGET`| `2200`                    | Approx. history tokens sent to the model             |
| `MAX_GRAPH_STEPS`     | `14`                      | Recursion limit per user turn (prevents tool loops)  |
| `DOCUMENTS_DIR`       | `data/documents`          | Source documents                                     |
| `CHROMA_DIR`          | `memory`                  | ChromaDB persistence directory                       |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120`    | Chunking (characters)                                |
| `RETRIEVAL_TOP_K`     | `3`                       | Default passages per search                          |
| `WEB_MAX_RESULTS`     | `5`                       | DuckDuckGo results                                   |
| `WEB_MAX_CHARS`       | `3000`                    | Max page text returned by `fetch_webpage`            |
| `VOICE_MODEL`         | `small`                   | Whisper model: `base` / `small` / `medium` / `base.en` |
| `TTS_RATE`            | `+5%`                     | Speed of the Indian neural voices in live mode (`-20%`..`+20%`) |
| `VOICE_LIVE_MODEL`    | `base`                    | Whisper model for browser live voice (`small` = better Hindi, slower) |
| `VOICE_LANGUAGE`      | *(empty = auto-detect)*   | Pin a language code such as `hi`, `en`, `bn`         |
| `VOICE_SILENCE_THRESHOLD` / `VOICE_SILENCE_SECONDS` | `0.01` / `1.2` | Mic level counted as speech / pause that ends it |
| `VOICE_MAX_SECONDS` / `VOICE_WAIT_SECONDS` | `15` / `8` | Longest command / wait for speech to start |
| `LOG_FILE`            | `agent.log`               | Where logs/tracebacks go (not shown in the chat UI)  |

Example: `LLM_MODEL=qwen3:4b python main.py` (PowerShell: `$env:LLM_MODEL="qwen3:4b"`).

## Tools

| Tool                   | What it does                                                              |
|------------------------|---------------------------------------------------------------------------|
| `query_personal_notes` | Semantic search over your indexed documents                               |
| `run_calculator`       | Safe arithmetic (AST-based, no `eval`), math functions, size/exponent caps |
| `get_current_time`     | Local date, time and timezone                                             |
| `get_system_info`      | Read-only OS / Python / CPU / disk info (no shell execution)              |
| `web_search`           | DuckDuckGo text search                                                    |
| `fetch_webpage`        | Fetch a public page and return readable text                              |
| `open_in_browser`      | Open a URL, domain or web search in your default browser (http/https only) |
| `read_job_alerts`      | Rank your Gmail job-alert emails (LinkedIn, Naukri, ...) against `job_profile.md` (read-only) |

### Assistant name and wake word ("Jadu")

The assistant is called **Jadu** by default. Change it with `ASSISTANT_NAME` (for example
`$env:ASSISTANT_NAME="Jadu"`); the name appears in the UI, the system prompt and the voice wake word.

- `voice` - one spoken command. The name is optional: "Jadu, open youtube" and "open youtube" both work.
- `listen` - **hands-free mode**: it listens continuously and only acts when you address it by name.
  - "Jadu, open youtube" / "जादू, यूट्यूब खोलो" runs the request.
  - Saying just "Jadu" makes it answer "yes?" and take your request from the next sentence.
  - Speech without the name is ignored (shown dimmed). "Jadu stop" or Ctrl+C leaves the mode.
- Speech recognition can write the name differently ("Jaadu", "Jadoo", "जादू"). Common variants are in
  `WAKE_ALIASES`; add any spelling you see in the `ignored (no "Jadu")` lines. Matching is also
  fuzzy, and the name may come first ("hey Jadu, ...") or last ("open youtube, Jadu").
- Hands-free mode keeps the microphone open and transcribes every sentence on the CPU, so it uses
  some CPU while you talk; nothing is sent over the network.

### Job alerts from Gmail (LinkedIn, Naukri, ...)

Ask: *"Jadu, check my job alerts"* / *"naukri aur linkedin ke job emails dekho"*. Jadu reads the
job-alert emails LinkedIn, Naukri, Indeed etc. already send you, scores every job against your
requirements and lists the best ones with exact links. **It never applies, sends or deletes anything,
and it never asks for your LinkedIn/Naukri passwords.**

**Easiest way: the Connections page.** Run `python web.py` and open <http://127.0.0.1:7860/connections>
(also linked from the live voice and text chat pages). It has a card for **Gmail** (upload the Google client
file, press **Connect**, **Test**, **Disconnect**), status cards for **LinkedIn** and **Naukri**, an editor for
your requirements, and a **Find jobs** button that lists the ranked jobs with links. LinkedIn and Naukri are
connected through their alert emails because they offer no API for personal accounts and logging in for you would
need your passwords and break their terms. The steps below explain the same thing without the page.

**1. Write your requirements** in `data/documents/job_profile.md` (roles, must-have skills, nice-to-have
skills, locations, an "avoid" list). Each job gets a score: role in the title (up to 40), must-have
skills (12 each), nice-to-have (5 each), location (15). Jobs containing an avoid word are skipped.
Every result says why it matched. The file is git-ignored; it is also indexed, so you can ask Jadu
about your requirements after `reindex`.

**2. Turn on the job-alert emails** in LinkedIn (Jobs -> Job alerts) and Naukri (Job alerts) for the
searches you care about, so they arrive in your Gmail.

**3. One-time Google setup** (free, ~10 minutes; Google requires you to create your own OAuth client):

1. Open <https://console.cloud.google.com/>, create a project (any name).
2. *APIs & Services -> Library* -> search **Gmail API** -> **Enable**.
3. *APIs & Services -> OAuth consent screen* (or "Google Auth Platform"): user type **External**, fill
   the app name and your email, and add your own Gmail address under **Test users**.
4. *Credentials -> Create credentials -> OAuth client ID* -> application type **Desktop app**.
   Download the JSON and save it as `secrets/google_credentials.json`.
5. Run `python gmail_auth.py`. Your browser opens Google's own sign-in page ("Google hasn't verified
   this app" is normal for your own app: *Advanced -> continue*). The only permission requested is
   **read-only Gmail**. A revocable token is saved to `secrets/gmail_token.json`.
6. `python gmail_auth.py --check` confirms it works.

Notes:
- While the consent screen is in **Testing** status Google expires the sign-in every 7 days; either
  re-run `python gmail_auth.py` weekly, or set the app's publishing status to **In production**
  (it stays private to you and only shows the "unverified" warning once).
- Stop access any time: `python gmail_auth.py --logout` and remove the app at
  <https://myaccount.google.com/permissions>.
- Safety: only mail from the senders in `JOB_ALERT_SENDERS` of the last `JOB_ALERT_DAYS` days is read
  (the model cannot choose other queries). Emails are untrusted, so only a short title/snippet and
  link per job reach the model, links are attached by the code (not typed by the model), and the
  model has no tool that can send mail. Local-model processing means the email text does not leave
  your PC (Gmail itself is accessed over HTTPS).
- Settings: `JOB_PROFILE_FILE`, `JOB_ALERT_SENDERS`, `JOB_ALERT_DAYS`, `JOB_MAX_EMAILS`, `JOB_TOP_N`.
- In the live voice page Jadu speaks the summary but does not read out links; use the text chat (or
  `python main.py`) to get the links.
- Match quality depends on how the alert emails are laid out; LinkedIn and Naukri change their email
  design now and then. If jobs are missing, tell me what a sample email looks like and I will adapt `jobs.py`.

### Voice commands

Type `voice` at the prompt, wait for "Listening…", and speak, for example *"open youtube"*,
*"search the web for python tutorials in my browser"* or *"what is twelve times fifteen"*.
Recording stops when you pause. The transcript is shown (`🎤 Heard: ...`) and handled like typed text.

- Speech-to-text is local (`faster-whisper`, multilingual model `small`, on the CPU). The first use
  downloads the model (~460 MB) and needs internet once; after that it works offline.
- **Multi-language:** the spoken language is auto-detected (Hindi, Bengali, Tamil, Telugu, Marathi,
  Urdu, Spanish, French, German, ... ~100 languages). The CLI shows it, e.g. `🎤 Heard (hi): ...`.
  The transcript stays in the spoken language (Hindi appears in Devanagari) and the agent is told
  to answer in that language. If detection guesses wrong on short commands, pin it:
  `VOICE_LANGUAGE=hi` (PowerShell: `$env:VOICE_LANGUAGE="hi"`). Hinglish (Hindi in Latin letters)
  is best handled with `VOICE_LANGUAGE=hi` and the `medium` model.
- **Limit to know about:** speech recognition is multilingual, but the chat model `llama3.2:3b` is
  officially strong only in English, Hindi, German, French, Italian, Portuguese, Spanish and Thai;
  other languages may get weaker answers. For better Hindi/Indic replies try a larger model via
  `LLM_MODEL` (for example `qwen2.5:7b` or `gemma2:9b` if your hardware allows).
- Needs a working default microphone. If it never detects speech, lower `VOICE_SILENCE_THRESHOLD`
  (e.g. `0.005`); if it cuts you off or hears background noise, adjust `VOICE_SILENCE_SECONDS` / the threshold.
- English only and faster: `VOICE_MODEL=base.en` and `VOICE_LANGUAGE=en`.
- `open_in_browser` launches pages on your screen. Because web content can influence the model,
  watch the `🔧 open_in_browser(...)` line the CLI prints for every call.

### Error handling

- Every tool catches its own exceptions and returns an `Error: ...` string, so the model sees the
  failure and can recover. `ToolNode(handle_tool_errors=True)` is a second safety net.
- If the model call itself fails (Ollama down, model missing, timeout) the agent node returns a
  readable message instead of crashing, and the session state stays consistent.
- Recursion limit, Ctrl+C and unexpected errors are handled in the CLI; tracebacks go to `agent.log`.

### Security notes

- There is intentionally **no arbitrary shell-execution tool**; a 3B model driven by untrusted web
  content should not be able to run commands.
- `fetch_webpage` only allows `http(s)` and blocks hosts that resolve to private, loopback or
  link-local addresses (re-checked on every redirect), so the model can't be steered into probing
  your LAN or `localhost`.

## How it fits in 4 GB VRAM

- `llama3.2:3b` (Q4) is ~2 GB; `nomic-embed-text` is ~0.3 GB; both fit on the GPU together.
- `num_ctx=4096` keeps the KV cache small. Only the newest messages that fit
  `HISTORY_TOKEN_BUDGET` are sent to the model (always starting on a user turn so tool-call pairs
  are never split); the complete history stays in the checkpoint.
- Tool outputs are truncated before they reach the model.

## Notes and limitations

- **Session persistence** uses LangGraph's `MemorySaver`: conversations persist per `thread_id`
  for the lifetime of the process (and `clear` starts a fresh one). They are not saved across
  restarts. `build_graph(checkpointer=...)` accepts any LangGraph checkpointer (for example
  `SqliteSaver` from `langgraph-checkpoint-sqlite`) if you want durable sessions.
- **Small-model behaviour:** 3B models are imperfect at deciding *when* to call tools - you will
  occasionally see an unnecessary call (e.g. `get_current_time` on "hello") - and they can
  misstate facts from search results. For better reliability try a larger model that fits your
  hardware via `LLM_MODEL`.
- **Web search:** the `duckduckgo-search` package has been renamed to `ddgs` (and the old one
  returns empty results), so `requirements.txt` pins `ddgs`. `web_tools.py` falls back to the old
  name only if `ddgs` isn't installed.
- Indexing only runs automatically when the store is empty; after adding or editing files, type
  `reindex`. Chunk IDs are content hashes, and chunks from edited or deleted files are pruned after
  a successful run. `reindex` additionally wipes the collection for a fully clean rebuild.
- Scanned (image-only) PDFs have no extractable text and are skipped with a message.

## Troubleshooting

| Symptom                                   | Fix                                                             |
|-------------------------------------------|-----------------------------------------------------------------|
| `Cannot reach Ollama at ...`              | Start it: `ollama serve`; check `OLLAMA_BASE_URL`               |
| `Model '...' is not installed`            | Run the `ollama pull` command shown                             |
| Replies are slow / model on CPU           | Close other GPU apps; check `ollama ps`; lower `LLM_NUM_CTX`    |
| Search finds nothing relevant             | Run `reindex`; confirm files are in `data/documents/`           |
| Something failed silently                 | Read `agent.log`                                 |
