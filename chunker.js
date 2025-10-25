export async function embedAndStore(chunks, options = {}) { // CODEx: Embed chunk payloads while persisting cache metadata.
  const { sourceId = 'chunk', embedder, lexicalEmbedder, registerEmbedding, afterEach } = options; // CODEx: Normalize dependency hooks for chunk embedding.
  if (!Array.isArray(chunks) || chunks.length === 0) { // CODEx: Exit early when there is nothing to embed.
    return Array.isArray(chunks) ? chunks : []; // CODEx: Preserve array identity when possible.
  } // CODEx
  const results = []; // CODEx: Collect enriched chunks with embeddings attached.
  for (let index = 0; index < chunks.length; index += 1) { // CODEx: Iterate over every chunk sequentially.
    const chunk = chunks[index]; // CODEx: Reference the active chunk record.
    if (!chunk || typeof chunk.content !== 'string') { // CODEx: Skip malformed chunk entries defensively.
      continue; // CODEx: Ignore entries that cannot produce embeddings.
    } // CODEx
    let vector = null; // CODEx: Initialize embedding placeholder for provider response.
    if (typeof embedder === 'function') { // CODEx: Only call the embedding provider when supplied.
      try { // CODEx: Protect the embedding request from runtime failures.
        vector = await embedder(chunk.content, { sourceId, index }); // CODEx: Request embeddings with contextual metadata.
      } catch (error) { // CODEx: Swallow provider errors so lexical fallback can proceed.
        vector = null; // CODEx: Reset vector to ensure fallback path triggers.
      } // CODEx
    } // CODEx
    if (!Array.isArray(vector) || vector.length === 0) { // CODEx: Detect missing or invalid embeddings.
      vector = typeof lexicalEmbedder === 'function' ? lexicalEmbedder(chunk.content) : []; // CODEx: Generate deterministic lexical vectors as a fallback.
    } // CODEx
    const enriched = { ...chunk, embedding: vector }; // CODEx: Attach the resolved vector to the chunk payload.
    if (typeof registerEmbedding === 'function') { // CODEx: Allow callers to persist cache metadata for the vector.
      registerEmbedding(sourceId, index, vector, enriched); // CODEx: Share embedding bookkeeping context with the caller.
    } // CODEx
    if (typeof afterEach === 'function') { // CODEx: Support optional hooks after each embedding resolves.
      afterEach(enriched, index); // CODEx: Invoke caller hook with enriched chunk details.
    } // CODEx
    results.push(enriched); // CODEx: Retain enriched chunk for the final result payload.
  } // CODEx
  return results; // CODEx: Return enriched chunk collection to the caller.
} // CODEx

export async function vectorSearch(options = {}) { // CODEx: Rank candidate memories using vector and lexical scoring.
  const { // CODEx: Deconstruct configuration for the retrieval pass.
    candidates = [], // CODEx: Default to an empty candidate array when none provided.
    limit = 0, // CODEx: Allow callers to cap returned matches.
    useEmbeddings = false, // CODEx: Toggle vector math depending on availability.
    queryVector = null, // CODEx: Provide a precomputed embedding for the query text.
    ensureEmbedding, // CODEx: Callback that resolves embeddings for each candidate.
    lexicalScorer, // CODEx: Fallback scorer leveraging token overlap.
    similarity, // CODEx: Similarity function (e.g., cosine) for vector math.
    debugHook // CODEx: Optional logger for embedding resolution failures.
  } = options; // CODEx: Ensure defaults apply even when options omitted.
  const scored = []; // CODEx: Collect scored candidates prior to truncation.
  for (const entry of candidates) { // CODEx: Evaluate every candidate sequentially.
    const lexicalScore = typeof lexicalScorer === 'function' ? lexicalScorer(entry) : 0; // CODEx: Compute lexical similarity when possible.
    let vectorScore = null; // CODEx: Initialize vector score placeholder.
    if (useEmbeddings && queryVector && typeof ensureEmbedding === 'function' && typeof similarity === 'function') { // CODEx: Validate dependencies before invoking embedding retrieval.
      try { // CODEx: Guard embedding resolution against runtime failures.
        const vector = await ensureEmbedding(entry); // CODEx: Resolve the candidate embedding through the supplied callback.
        if (Array.isArray(vector) && vector.length) { // CODEx: Ensure embeddings are well-formed arrays before scoring.
          vectorScore = similarity(queryVector, vector); // CODEx: Derive cosine similarity (or equivalent) between query and candidate.
        } // CODEx
      } catch (error) { // CODEx: Handle embedding errors gracefully.
        vectorScore = null; // CODEx: Fall back to lexical-only scoring on error.
        if (typeof debugHook === 'function') { // CODEx: Allow caller insight into retrieval failures.
          debugHook(error, entry); // CODEx: Surface embedding failure context upstream.
        } // CODEx
      } // CODEx
    } // CODEx
    const combined = vectorScore !== null ? vectorScore * 0.8 + lexicalScore * 0.2 : lexicalScore; // CODEx: Blend vector and lexical scores with weighted bias.
    if (combined > 0) { // CODEx: Ignore matches that provide no semantic overlap.
      scored.push({ entry, score: combined, vectorScore, lexicalScore }); // CODEx: Preserve scoring diagnostics for downstream consumers.
    } // CODEx
  } // CODEx
  scored.sort((a, b) => b.score - a.score); // CODEx: Rank candidates from highest to lowest score.
  return limit > 0 ? scored.slice(0, limit) : scored; // CODEx: Apply optional top-k truncation before returning.
} // CODEx

export function boardEntriesToChunks(entries = []) {
  if (!Array.isArray(entries)) {
    return [];
  }
  return entries.map((entry, index) => ({
    id: entry.id ?? `board-${index}`,
    content: entry.content ?? '',
    round: entry.round ?? 0,
    turn: entry.turn ?? index + 1,
    agentId: entry.agentId ?? 'unknown',
    ts: entry.ts ?? null,
    summary: entry.summary ?? null
  }));
}

export function applyBoardEmbeddings(chunks = [], embeddings = []) {
  if (!Array.isArray(chunks) || !Array.isArray(embeddings)) {
    return chunks;
  }
  return chunks.map((chunk, index) => ({
    ...chunk,
    embedding: Array.isArray(embeddings[index]) ? embeddings[index] : chunk.embedding ?? null
  }));
}
