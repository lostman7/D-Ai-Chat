# SAM – Memory-Augmented Chat

SAM is a browser-based chat workspace that layers a floating working memory, long-term storage, and lightweight retrieval over any OpenAI-compatible chat endpoint (including [LM Studio](https://lmstudio.ai/)). The UI lives entirely in the browser, so you can point it at a locally hosted model, experiment with memory budgets, and even let two agents debate each other.

## Key Features

- **Memory governance** – Choose a floating-memory budget (50&nbsp;MB – 200&nbsp;MB). When the live buffer overflows, messages are archived to IndexedDB automatically and annotated in the transcript.
- **Floating memory workbench** – Inspect what’s currently “on the desk,” pin critical turns so they never get trimmed, manually archive unneeded messages, reload RAG archives on demand, and watch the live counters for recovered snapshots, total footprint, retrieval activity, and the most recent autosave.
- **Sliding options drawer** – Use the edge handle (‹/›) to pull the configuration hub into view, set memory budgets, swap model providers, and paste API keys without scrolling back to the top of a long transcript.
- **Global status dock** – A floating header keeps Model A/Model B readiness, floating-memory usage, system RAM/GPU notes, diagnostics, and mode toggles within reach no matter how long the transcript grows.
- **Workspace chooser** – Land on a lightweight launcher that lets you decide between human chat and the dual-agent arena before any UI loads.
- **Retrieval-augmented prompting** – Pull relevant memories back into context with a click. A cosine-similarity search runs across both floating and archived logs to assemble reference snippets for the next request.
- **Configurable model bridge** – Wire up Model A for the main chat and optionally enable a distinct Model B for the arena. Each can inherit curated presets (LM Studio, Ollama, OpenRouter, OpenAI, Groq, Together, Mistral, Perplexity, Fireworks, DeepSeek, xAI, Anthropic, Google) or point at your own endpoint, and everything persists in the local storage layer (browser `localStorage` or the standalone shell’s disk-backed store).
- **Speech in / speech out** – Dictate messages with the built-in browser speech-recognition API (Chrome/WebKit) and auto-speak assistant replies with a curated TTS pipeline (browser voices, Piper, Coqui XTTS, Bark, or ElevenLabs).
- **AMD-friendly voice presets** – Pick from community-rated speech engines that run well on AMD hardware, set server/voice credentials, and use the **Test voice** button to confirm playback.
- **Dual-agent arena** – Spin up two personas, prefill the persistent-memory debate seed (or add your own), route each model through different providers if you like, and watch turn-badged exchanges roll just like a normal chat while the SAMs trade unlimited volleys (or step through one at a time).
- **Reflex mode & entropy meter** – Let SAM periodically summarize what it has learned every _N_ turns (toggle in **Reflex & telemetry**) while the entropy meter highlights when debates are looping versus exploring fresh territory.
- **Conversation exporting** – Download `.txt` transcripts of the main chat or arena debates for offline analysis.
- **Replay phaser CLI** – Run `npm run replay ./rag/export.txt` to colorize transcripts in your terminal, with role-aware highlighting, equation pops, and a “new term” detector so you can skim long debates quickly.
- **Turn-numbered transcripts** – Every user/assistant pair (and each arena volley) receives a turn index so you can point SAM back to “turn 42” in a hurry.
- **Built-in diagnostics** – Run a quick health check from the header to confirm API connectivity, memory pressure, voice configuration, and dual-agent status.
- **Automatic RAG snapshots** – Every turn is mirrored into a `rag-logs` IndexedDB store. SAM now hydrates the floating buffer with archived chunks on load, saves every 30-message block with a timestamped checkpoint, and the arena still auto-saves its log every two minutes so long-form conversations are ready for retrieval without manual exports.
- **Manifest-driven RAG hydration** – The **Load RAG archives** button also ingests anything listed in `rag/manifest.json` or `rag/archives/manifest.json` (Markdown, JSON/JSONL, plain text, or PDF). Successfully loaded sources light up the new RAG status pill in the header so you always know when memories are synced.
- **File chunker system** – Automatically splits large RAG files into smaller chunks for better retrieval. Configure chunk size and overlap in the options panel, then scan and chunk files. Chunks are stored in `rag/chunked/` while originals move to `rag/unchunked/`. Visual status indicators show processing state (🔄 processing, 🟡 ready to chunk, 🟢 idle).
- **Custom backdrops** – Drop in an image URL or upload your own wallpaper to give SAM a new vibe; the gradient overlay keeps transcripts legible while the status dock floats above everything.
- **Debug console & run logs** – Enable the debug toggle in the options drawer to monitor live process states, capture startup/shutdown/error notes, and export timestamped run logs (persisted in IndexedDB and downloadable into the `logs/` folder).

## Getting Started

### Prerequisites

- **Node.js 18+** (only needed if you want to serve the page through Vite during development).
- **An OpenAI-compatible chat endpoint** – e.g. LM Studio’s local server (`http://localhost:1234/v1/chat/completions`) or a hosted OpenAI/compatible API key.

### Local Development

```bash
npm install
npm run dev
```

Vite will print a local URL (typically `http://localhost:5173`). Open it in a modern Chromium-based browser for the best speech-recognition support.

### Static analysis & smoke checks

Run the project self-check to confirm the UI scaffolding and scripts are wired correctly:

```bash
npm run lint
```

The check runs a Node.js syntax verification on `app.js` and ensures the UI contains the required configuration controls (options toggle, provider picker, floating memory workbench, etc.).

### Production Preview

You can also double-click `index.html` and run the app straight from disk. Voice input may require `https:` depending on the browser.

### Standalone desktop shell

Prefer an app-like shell that keeps its own storage separate from your browser? Install dependencies and launch the Electron wrapper:

```bash
npm install
npm run standalone
```

The shell loads the local `index.html`, exposes the same UI, and persists configuration/chunk data inside Electron’s `userData` directory. Set `STANDALONE_DEVTOOLS=1` when running the script if you want the developer tools to pop automatically.

## Using SAM

1. **Pick a workspace** – Use the landing overlay to choose between **Chat with SAM** (human ↔ SAM) or **Dual-agent arena** (SAM debating itself). You can swap modes later with the header shortcuts. The chunker status indicator shows processing state (🔄 processing, 🟡 ready to chunk, 🟢 idle).
2. **Configure Model A** – Pull the edge handle (‹/›) to open the sliding drawer and visit **Model workbench**. Pick a provider preset or enter the endpoint URL/model manually, paste any API key, tweak the system prompt, and set the temperature plus the **Max response tokens** guard (the input automatically respects the provider's context window). Click **Save model settings** when you're ready. This connection powers human chat and also feeds Model B when you leave it set to "Share Model A settings."
3. **Tune memory & retrieval** – Use the slider to pick a floating-memory size. Adjust how many turns are kept in the immediate context and how many retrieved memories are inserted per prompt (`0` makes retrieval unlimited so SAM can pull as many matches as exist). Tap **Load RAG archives** whenever you want to rehydrate fresh checkpoints _and_ any manifest-listed files under `rag/` into floating memory.
4. **Set up file chunking** – In the **File chunker** section, adjust chunk size (100-2000 tokens) and overlap (0-200 tokens). Click **Scan** to find files in `rag/` that need chunking. Click **Chunk all files** to process them - originals move to `rag/unchunked/` and chunks save to `rag/chunked/`. Use **Clear chunks** to remove all chunked/unchunked files. The scanner now zeroes in on plain-text sources (`.txt`, `.md`, `.markdown`) and structured notes (`.json`, `.jsonl`) no matter whether they live on disk or in the RAM-disk cache.
5. **Curate the floating workbench** – Pin important turns so they stay in RAM, archive anything you don't need, and watch the live counter (plus the per-model retrieval stats) to see how aggressively each SAM is consulting long-term memory.
5. **Tune the voice** – In **Speech synthesis**, choose a preset (browser, Piper, Coqui XTTS, Mimic 3, F5-TTS, Bark, or ElevenLabs), supply any required server URL or API key, then click **Test voice** to confirm audio.
6. **Chat normally** – Type or dictate a message and hit **Send**. SAM stores the turn in floating memory, retrieves relevant context, calls your configured endpoint for a reply, and keeps both transcript views auto-following the newest exchange.
7. **Promote archived memories** – Click **Promote relevant memories** to describe what you’d like SAM to recall. Matching snippets re-enter the floating buffer.
8. **Let SAMs debate** – Switch to the arena, flip **Enable Model B** if you want a second brain, and configure the personas for Model A and Model B. When Model B stays off, only the primary Model A settings remain visible so the drawer isn’t cluttered. Each model can inherit the primary connection or override it with its own provider preset/endpoint. Adjust the starter conversation (preloaded with the persistent-memory debate), then press **Start** to watch the transcript populate in real time with turn badges for SAM-A and SAM-B. The status dock will show whether each model is local (RAM/VRAM) or cloud-hosted. Leave the turn limit blank for an endless back-and-forth or enter a number if you want an automatic stop, and use the **Dual turn delay** field to give reasoning-heavy models up to 10 minutes of breathing room before the partner’s next reply auto-starts. The Reflex banner will confirm when summaries are queued, and the entropy meter glows red when the debate starts looping so you can intervene or trigger a manual summary.
10. **Run diagnostics** – Hit **Run diagnostics** in the chat header to probe the configured endpoint, review floating-memory pressure, and confirm speech/arena status.
11. **Personalize & export transcripts** – Open **Appearance** to paste a background URL, upload an image, or revert to the default gradient; custom artwork now applies instantly across both the browser and standalone shells. Use **Export conversation log** in chat or **Export transcript** in the arena footer whenever you need a plain-text copy of the latest exchanges. Want a quick colorized review? Run `npm run replay` against the exported file to open a role-aware, equation-highlighting replay in your terminal.
12. **Monitor logs & debug state** – In **Debug & Logs**, toggle the debug console to surface live process statuses, capture startup/shutdown/error notes, and export timestamped run logs (saved to IndexedDB and downloaded into the `logs/` folder).

### Provider presets & API keys

- **LM Studio / Ollama** – Local-first presets that auto-fill loopback endpoints so you can start with TinyLlama-sized models and scale up.
- **OpenRouter / OpenAI / Groq / Together** – Cloud bridges that require API keys; SAM highlights missing credentials so you can paste them before saving.
- **Mistral / Perplexity / Fireworks / DeepSeek** – Additional OpenAI-compatible hubs with generous context windows. Select the preset and drop in the corresponding API key to explore reasoning-optimized checkpoints.
- **xAI (Grok)** – Direct access to xAI's Grok models. Requires an xAI API key from x.ai for models like Grok-2.
- **Anthropic (Claude)** – Official Anthropic API access to Claude models. Requires an Anthropic API key for models like Claude 3.5 Sonnet.
- **Google (Gemini)** – Google's Gemini models via AI Studio API. Requires a Google AI API key for models like Gemini 1.5 Pro.
- **Custom** – Select this when pointing SAM at gateways such as OpenRouter proxies, LM Studio over LAN, or experimental backends. Fill in any extra headers using your reverse proxy of choice.

Switching presets updates the endpoint/model fields immediately while keeping your existing API key intact so you can hop between providers without retyping secrets.

## Speech Synthesis Presets

SAM ships with a configurable speech stack so you can run high-quality voices on AMD hardware or fall back to built-in browser speech. The selector shows community feedback and recommended defaults for each engine.

| Preset | Why it shines | Default server | Setup notes |
| --- | --- | --- | --- |
| Browser voice (Web Speech API) | Zero configuration fallback that uses the operating system voices. | — | Works everywhere; pick a voice from the browser catalog if one is available. |
| Piper • en_US-amy-low | 4.7★ Mycroft-community rating, fast inference, AMD-friendly via `onnxruntime-rocm`. | `http://localhost:5002` | Run `piper-tts --http 0.0.0.0:5002 --voice en_US-amy-low` and drop other Piper voices into the `voices/` folder. |
| Coqui XTTS v2 | 4.8★ multilingual, supports voice cloning and DirectML/ROCm acceleration. | `http://localhost:8021` | Launch with `tts --model XTTS-v2 --port 8021` and provide a reference audio clip for cloning if desired. |
| Mycroft Mimic 3 | 4.6★ open-source alternative with pronunciation dictionaries and light footprint. | `http://localhost:59125` | Serve with `mimic3-server --host 0.0.0.0 --port 59125 --voice en_US/mimic3_low`. |
| F5-TTS | 4.5★ community favorite for expressive, easily fine-tuned voices. | `http://localhost:8088` | Use `f5-tts-server --host 0.0.0.0 --port 8088` or the Text Generation WebUI plugin. |
| Suno Bark-small (AMD build) | Expressive narration with emotional cues; good community ROCm builds. | `http://localhost:5005` | Use an AMD-capable Bark HTTP server (`bark-server --host 0.0.0.0 --port 5005`). Select a speaker such as `en_speaker_6`. |
| ElevenLabs (cloud) | 4.9★ quality, instant responses, massive voice library. | `https://api.elevenlabs.io/v1/text-to-speech` | Paste your ElevenLabs voice ID and API key. Works even when you can’t host a local engine. |

Pick a preset, fill in the server URL/voice ID (defaults populate automatically), and toggle **Auto speak replies** to keep SAM narrating every assistant turn.

## Optional Backend Dependencies

If you want to host Piper, Coqui XTTS, Bark, or an API wrapper locally, install the optional Python stack listed in [`requirements.txt`](requirements.txt). It pins FastAPI, Piper, Coqui TTS, and AMD-friendly ONNX Runtime builds (ROCm/DirectML) so you can spin up compatible HTTP servers quickly:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pick the ONNX runtime that matches your platform (ROCm for Linux/AMD GPUs, DirectML for Windows) before starting your TTS server.

## Data Storage

- **Floating memory** – Maintained in JavaScript with an approximate byte budget.
- **Long-term memory** – All turns are persisted in the browser’s IndexedDB (`chatDatabase.messages`). Archived messages are labelled in the UI and remain searchable.
- **RAG snapshots** – Chat and arena transcripts are mirrored into IndexedDB (`chatDatabase.rag-logs`), every 30-message chunk is checkpointed with a timestamped record, and the arena auto-saves its running debate every two minutes so the retrieval layer and the `rag/` workspace always have fresh logs. On load SAM hydrates the floating buffer with those archived slices before you even send the first prompt, and you can tap **Load RAG archives** in the drawer at any time to pull the latest snapshots and any manifest-listed files under `rag/` (watch the new header pill flip to "RAG: Loaded" when synchronization completes).
- **Configuration** – Saved in persistent storage under the `sam-config` key (browser `localStorage` or the standalone shell’s store).
- **Pinned turns** – Stored alongside configuration data under `sam-pinned-messages` so your curated floating context survives reloads.
- **Run logs** – Captured in IndexedDB (`chatDatabase.sam-logs`) and exportable from **Debug & Logs**; downloads land in the repository’s `logs/` folder by default.

> ⚠️ All data stays on your machine. In the browser, clearing site data or switching browsers resets memories unless you export them first. The standalone shell keeps its state in Electron’s `userData` directory.

See [`rag/README.md`](rag/README.md) for tips on inspecting and reusing the automatically captured RAG bundles.

## Notes & Limitations

- Speech recognition relies on the experimental Web Speech API and currently works best in Chromium-based browsers.
- The retrieval scorer is a lightweight cosine similarity over token counts; for production workloads you may want to swap in embedding-based search.
- No backend services are included. Provide your own endpoint that implements the [Chat Completions](https://platform.openai.com/docs/api-reference/chat/create) interface.
- SAM defaults to a 512-token cap per reply to keep small-context models (e.g., LM Studio TinyLlama) from overflowing. The **Max response tokens** field now adjusts its ceiling to match the selected provider (LM Studio/OpenAI/OpenRouter up to ~262k tokens, Groq ~32k, etc.), so you can safely raise or lower the guard without overshooting the backend’s context window.
- Some servers ignore `max_tokens`; SAM now trims any overlong reply on the fly, stamps the message with a **trimmed** badge, and logs a system notice so you know the guard intervened. When a reasoning-capable model returns chain-of-thought traces, SAM now preserves both the reasoning and the final answer in the transcript so nothing is lost.

Happy tinkering with persistent AI memories!
