# RAG Snapshots

SAM automatically mirrors both the human chat transcript and the dual-agent arena dialogue into the browser's IndexedDB under the `rag-logs` object store. Each time a message is added, a snapshot entry is updated with the latest content, the arena runs an automatic autosave every two minutes, and every 30-message block is captured as its own timestamped checkpoint. On reload SAM hydrates the floating buffer with these archived slices so memories are available before the first new prompt is sent. The floating memory workbench now surfaces how many RAG snapshots were loaded and when the most recent autosave fired so you can confirm the pipeline is up to date without opening dev tools.

Need to pull in fresh checkpoints manually? Click **Load RAG archives** in the options drawer to hydrate the floating buffer on demand and watch the snapshot counters update live. The reload pass now looks in two places:

1. **IndexedDB snapshots** – the browser-native `rag-logs` store that captures rolling chat and arena transcripts.
2. **On-disk manifest files** – any entries listed in `rag/manifest.json` or `rag/archives/manifest.json`. These files can point to Markdown, plain text, JSON/JSONL, or PDF documents inside the `rag/` tree. Each source is scanned, converted into memory snippets, and pushed into the floating buffer without waiting for a new prompt.

Every static entry should include at least a `path`. Optional fields like `role`, `mode`, `label`, and `pinned` let you control how the memory shows up inside SAM.

```json
{
  "entries": [
    {
      "id": "foundational-notes",
      "label": "Foundational SAM notes",
      "path": "notes/primer.md",
      "role": "system",
      "mode": "chat",
      "pinned": true
    }
  ]
}
```

Place arena transcripts under `rag/archives/` and add them to `rag/archives/manifest.json` to surface debates automatically. PDFs are supported through the bundled pdf.js helper; other formats are treated as plain text.

## What gets stored

- **Type** – `snapshot` for the rolling latest state, or `checkpoint` for the 30-message bundles that accumulate in chronological order.
- **Mode** – `chat` or `arena`, so you can tell whether the entry came from a human ↔ SAM session or the self-debate arena.
- **Seed** – For arena runs, the starter conversation that kicked off the debate (handy when you rotate seeds).
- **Messages** – A trimmed list of recent turns (up to 400 for chat, 200 for arena) including timestamps, speaker labels, pinned status, and the shared turn numbers that power the on-screen badges.
- **Updated timestamp** – When the snapshot was last refreshed.

## Inspecting the data

Open your browser's developer tools and head to the **Application › IndexedDB** tab. Expand `chatDatabase` and you will find `rag-logs`. Each key maps to a session ID such as `arena-session-2024-05-20T18-44-11-123Z`.

You can right-click an entry to export it as JSON, or iterate over the store in the console:

```js
const request = indexedDB.open('chatDatabase');
request.onsuccess = () => {
  const db = request.result;
  const tx = db.transaction(['rag-logs'], 'readonly');
  const store = tx.objectStore('rag-logs');
  store.getAll().onsuccess = (event) => {
    console.log(event.target.result);
  };
};
```

Because snapshots and checkpoints stay in the browser, you can clear them by removing site data. If you want an on-disk copy, export the object store contents or use the built-in **Export conversation log** / **Export transcript** buttons for a text version. Static manifest files live under `rag/` and can be committed to source control for sharing long-term memories with teammates.
