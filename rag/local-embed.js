// CODEx: Define constants representing supported embedding providers.
export const EMBEDDING_PROVIDERS = Object.freeze({ // CODEx: Freeze to prevent runtime mutation.
  AUTO: 'auto', // CODEx: Automatic detection mode.
  OPENAI: 'openai', // CODEx: Remote OpenAI-style HTTP embedding provider.
  LM_STUDIO: 'lmstudio', // CODEx: Local LM Studio embedding endpoint.
  OLLAMA: 'ollama' // CODEx: Local Ollama embedding endpoint.
}); // CODEx: Close provider constant definition.

const DEFAULT_LM_STUDIO_BASE = 'http://localhost:1234'; // CODEx: Fallback LM Studio base URL.
const DEFAULT_OLLAMA_BASE = 'http://localhost:11434'; // CODEx: Fallback Ollama base URL.
const OPENAI_EMBEDDING_PATH = 'https://api.openai.com/v1/embeddings'; // CODEx: Default remote embedding endpoint.

function normalizeBaseUrl(candidate) { // CODEx: Ensure base URLs exclude trailing slashes.
  if (!candidate) { // CODEx: Guard nullish inputs.
    return ''; // CODEx: Return empty string when not provided.
  } // CODEx
  try { // CODEx: Attempt structured parsing first.
    const origin = typeof window !== 'undefined' ? window.location.origin : undefined; // CODEx: Derive optional base origin.
    const parsed = new URL(candidate, origin); // CODEx: Normalize using URL constructor.
    parsed.hash = ''; // CODEx: Drop hash fragments.
    parsed.pathname = ''; // CODEx: Strip path components to isolate origin.
    parsed.search = ''; // CODEx: Remove query parameters for base resolution.
    const normalized = parsed.toString().replace(/\/$/, ''); // CODEx: Strip trailing slash from final string.
    return normalized; // CODEx: Return normalized base URL.
  } catch (error) { // CODEx: Fall back to manual sanitation when URL constructor fails.
    const sanitized = String(candidate).split(/[?#]/)[0]; // CODEx: Remove query/hash segments manually.
    const protocolSplit = sanitized.split('://'); // CODEx: Separate protocol from host/path.
    if (protocolSplit.length === 2) { // CODEx: Preserve protocol and host only.
      const [protocol, rest] = protocolSplit; // CODEx: Deconstruct protocol and remainder.
      const host = rest.split('/')[0]; // CODEx: Extract hostname with optional port.
      return `${protocol}://${host}`.replace(/\/$/, ''); // CODEx: Reconstruct origin.
    }
    return sanitized.replace(/\/.*$/, '').replace(/\/$/, ''); // CODEx: Drop path segments for schemeless values.
  } // CODEx
} // CODEx

function deriveCandidateBases(options = {}) { // CODEx: Create ordered list of probe bases.
  const { embeddingEndpoint, modelEndpoint } = options; // CODEx: Destructure hints for reuse.
  const bases = new Set(); // CODEx: Use Set to avoid duplicates.
  if (embeddingEndpoint) { // CODEx: Include explicit embedding endpoint first.
    bases.add(normalizeBaseUrl(embeddingEndpoint)); // CODEx: Normalize before insertion.
  } // CODEx
  if (modelEndpoint) { // CODEx: Include main model endpoint as secondary hint.
    bases.add(normalizeBaseUrl(modelEndpoint)); // CODEx: Normalize model endpoint base.
  } // CODEx
  bases.add(DEFAULT_LM_STUDIO_BASE); // CODEx: Ensure LM Studio fallback candidate.
  bases.add(DEFAULT_OLLAMA_BASE); // CODEx: Ensure Ollama fallback candidate.
  return Array.from(bases).filter(Boolean); // CODEx: Return unique, non-empty candidates.
} // CODEx

async function probeLmStudio(base, fetchImpl) { // CODEx: Determine if LM Studio responds at the base URL.
  const target = `${base}/api/models`; // CODEx: LM Studio exposes /api/models listing.
  const response = await fetchImpl(target, { method: 'GET' }); // CODEx: Issue probe request.
  if (!response.ok) { // CODEx: Non-200 responses fail detection.
    throw new Error(`LM Studio probe ${response.status}`); // CODEx: Signal detection failure.
  } // CODEx
  await response.arrayBuffer(); // CODEx: Consume body to avoid leaked locks.
  return true; // CODEx: Affirm detection success.
} // CODEx

async function probeOllama(base, fetchImpl) { // CODEx: Determine if Ollama responds at the base URL.
  const target = `${base}/api/tags`; // CODEx: Ollama exposes /api/tags listing.
  const response = await fetchImpl(target, { method: 'GET' }); // CODEx: Issue probe request.
  if (!response.ok) { // CODEx: Non-200 responses fail detection.
    throw new Error(`Ollama probe ${response.status}`); // CODEx: Signal detection failure.
  } // CODEx
  await response.arrayBuffer(); // CODEx: Consume body to free stream.
  return true; // CODEx: Affirm detection success.
} // CODEx

export async function detectEmbeddingService(options = {}) { // CODEx: Resolve active embedding provider.
  const { // CODEx: Extract detection parameters.
    preference = EMBEDDING_PROVIDERS.AUTO, // CODEx: Requested provider preference.
    embeddingEndpoint = '', // CODEx: Optional configured embedding endpoint.
    modelEndpoint = '', // CODEx: Optional primary model endpoint.
    fetchImpl = fetch, // CODEx: Allow caller to supply custom fetch.
    logger = console // CODEx: Logging sink for diagnostics.
  } = options; // CODEx: Close destructuring definition.

  const normalizedPreference = (preference || EMBEDDING_PROVIDERS.AUTO).toLowerCase(); // CODEx: Normalize preference text.
  const candidates = deriveCandidateBases({ embeddingEndpoint, modelEndpoint }); // CODEx: Compute probe order.

  async function resolveAuto() { // CODEx: Auto-detection helper.
    for (const base of candidates) { // CODEx: Iterate through candidate bases.
      if (!base) { // CODEx: Skip empty candidates defensively.
        continue; // CODEx: Continue with next candidate.
      } // CODEx
      try { // CODEx: Attempt LM Studio detection first.
        await probeLmStudio(base, fetchImpl); // CODEx: Probe LM Studio endpoint.
        logger.info?.(`[RAG] Provider = LMStudio (${base})`); // CODEx: Log detection success.
        return { provider: EMBEDDING_PROVIDERS.LM_STUDIO, baseUrl: base }; // CODEx: Return detection result.
      } catch (lmError) { // CODEx: Swallow LM Studio failure.
        try { // CODEx: Attempt Ollama detection next.
          await probeOllama(base, fetchImpl); // CODEx: Probe Ollama endpoint.
          logger.info?.(`[RAG] Provider = Ollama (${base})`); // CODEx: Log detection success.
          return { provider: EMBEDDING_PROVIDERS.OLLAMA, baseUrl: base }; // CODEx: Return detection result.
        } catch (ollamaError) { // CODEx: Continue probing other candidates.
          logger.debug?.('[RAG] Local probe skipped', { base, lmError, ollamaError }); // CODEx: Provide verbose trace.
        } // CODEx
      } // CODEx
    } // CODEx
    logger.info?.('[RAG] Provider = OpenAI (remote fallback)'); // CODEx: Default to remote provider logging.
    return { provider: EMBEDDING_PROVIDERS.OPENAI, baseUrl: normalizeBaseUrl(embeddingEndpoint) || '' }; // CODEx: Return fallback result.
  } // CODEx

  if (normalizedPreference === EMBEDDING_PROVIDERS.OPENAI) { // CODEx: Remote provider forced by config.
    logger.info?.('[RAG] Provider = OpenAI (forced)'); // CODEx: Log forced selection.
    return { provider: EMBEDDING_PROVIDERS.OPENAI, baseUrl: normalizeBaseUrl(embeddingEndpoint) || '' }; // CODEx: Return remote selection.
  } // CODEx
  if (normalizedPreference === EMBEDDING_PROVIDERS.LM_STUDIO) { // CODEx: LM Studio forced by config.
    const base = normalizeBaseUrl(embeddingEndpoint || modelEndpoint || DEFAULT_LM_STUDIO_BASE); // CODEx: Derive base fallback.
    logger.info?.(`[RAG] Provider = LMStudio (forced ${base})`); // CODEx: Log forced selection.
    return { provider: EMBEDDING_PROVIDERS.LM_STUDIO, baseUrl: base }; // CODEx: Return forced LM Studio.
  } // CODEx
  if (normalizedPreference === EMBEDDING_PROVIDERS.OLLAMA) { // CODEx: Ollama forced by config.
    const base = normalizeBaseUrl(embeddingEndpoint || modelEndpoint || DEFAULT_OLLAMA_BASE); // CODEx: Derive base fallback.
    logger.info?.(`[RAG] Provider = Ollama (forced ${base})`); // CODEx: Log forced selection.
    return { provider: EMBEDDING_PROVIDERS.OLLAMA, baseUrl: base }; // CODEx: Return forced Ollama.
  } // CODEx

  return resolveAuto(); // CODEx: Execute auto-detection when preference is AUTO.
} // CODEx

function ensureJsonResponse(response) { // CODEx: Validate JSON MIME type heuristically.
  const contentType = response.headers.get('content-type') || ''; // CODEx: Retrieve content type header.
  if (!/json/i.test(contentType)) { // CODEx: Heuristic for JSON responses.
    throw new Error(`Unexpected content-type: ${contentType}`); // CODEx: Signal invalid response.
  } // CODEx
} // CODEx

function normalizeVectorCandidate(candidate) { // CODEx: Convert raw payloads into numeric arrays.
  if (Array.isArray(candidate)) { // CODEx: Accept direct arrays.
    return candidate.map((value) => Number.parseFloat(value) || 0); // CODEx: Coerce to floats.
  } // CODEx
  return null; // CODEx: Unsupported shape returns null.
} // CODEx

export async function requestEmbeddingVector(options = {}) { // CODEx: Retrieve embeddings from the active provider.
  const { // CODEx: Extract request parameters.
    provider = EMBEDDING_PROVIDERS.OPENAI, // CODEx: Active provider identifier.
    baseUrl = '', // CODEx: Base endpoint for local providers.
    endpointOverride = '', // CODEx: Optional explicit endpoint override.
    model = 'text-embedding-3-large', // CODEx: Embedding model identifier.
    text, // CODEx: Source text to embed.
    apiKey = '', // CODEx: Optional API key for remote providers.
    fetchImpl = fetch, // CODEx: Custom fetch injection.
    logger = console, // CODEx: Logging facility.
    signal // CODEx: AbortSignal for cancellation.
  } = options; // CODEx: Close destructuring definition.

  if (typeof text !== 'string' || !text.trim()) { // CODEx: Validate input text.
    throw new Error('Embedding text must be a non-empty string'); // CODEx: Reject invalid inputs.
  } // CODEx

  const trimmed = text.trim(); // CODEx: Normalize whitespace.
  const headers = { 'Content-Type': 'application/json' }; // CODEx: Default JSON headers.
  if (apiKey && provider === EMBEDDING_PROVIDERS.OPENAI) { // CODEx: Apply Authorization header for remote provider.
    headers.Authorization = `Bearer ${apiKey}`; // CODEx: Attach bearer token.
  } // CODEx

  const fetchOptions = { method: 'POST', headers, signal }; // CODEx: Baseline fetch options.

  if (provider === EMBEDDING_PROVIDERS.LM_STUDIO) { // CODEx: LM Studio request handling.
    const normalizedBase = normalizeBaseUrl(baseUrl || endpointOverride || DEFAULT_LM_STUDIO_BASE); // CODEx: Derive LM Studio base path.
    let target = `${normalizedBase}/api/generate`; // CODEx: Default LM Studio embedding endpoint.
    if (endpointOverride && !/\/api\/generate$/i.test(endpointOverride)) { // CODEx: Detect unexpected overrides when provided.
      logger.warn?.('[RAG] Unexpected endpoint → retrying with /api/generate'); // CODEx: Surface correction log.
    } // CODEx
    const payload = { prompt: trimmed, stream: false }; // CODEx: LM Studio generate payload.
    const response = await fetchImpl(target, { ...fetchOptions, body: JSON.stringify(payload) }); // CODEx: Execute LM Studio request.
    if (!response.ok) { // CODEx: Validate response status.
      throw new Error(`LM Studio embedding HTTP ${response.status}`); // CODEx: Propagate failure.
    } // CODEx
    ensureJsonResponse(response); // CODEx: Validate JSON content type.
    const data = await response.json(); // CODEx: Parse JSON body.
    const vector = normalizeVectorCandidate(data?.embedding || data?.data?.[0]?.embedding); // CODEx: Extract embedding vector.
    if (!vector) { // CODEx: Ensure embedding exists.
      throw new Error('LM Studio embedding missing'); // CODEx: Signal extraction failure.
    } // CODEx
    return vector; // CODEx: Return numeric vector.
  } // CODEx

  if (provider === EMBEDDING_PROVIDERS.OLLAMA) { // CODEx: Ollama request handling.
    const normalizedBase = normalizeBaseUrl(endpointOverride || baseUrl || DEFAULT_OLLAMA_BASE); // CODEx: Derive Ollama base path.
    const primary = `${normalizedBase}/api/embed`; // CODEx: Primary Ollama embedding endpoint.
    const payload = { model, input: trimmed }; // CODEx: Ollama embed payload.
    const response = await fetchImpl(primary, { ...fetchOptions, body: JSON.stringify(payload) }); // CODEx: Execute Ollama request.
    if (!response.ok) { // CODEx: Handle errors.
      throw new Error(`Ollama embedding HTTP ${response.status}`); // CODEx: Propagate failure.
    } // CODEx
    ensureJsonResponse(response); // CODEx: Validate JSON content type.
    const data = await response.json(); // CODEx: Parse JSON body.
    const vector = normalizeVectorCandidate(data?.embedding || data?.data?.[0]?.embedding); // CODEx: Extract embedding vector.
    if (!vector) { // CODEx: Ensure vector present.
      throw new Error('Ollama embedding missing'); // CODEx: Propagate extraction failure.
    } // CODEx
    return vector; // CODEx: Return numeric vector.
  } // CODEx

  const endpoint = endpointOverride || OPENAI_EMBEDDING_PATH; // CODEx: Resolve remote embedding endpoint.
  const payload = { model, input: [trimmed] }; // CODEx: OpenAI-compatible payload.
  const response = await fetchImpl(endpoint, { ...fetchOptions, body: JSON.stringify(payload) }); // CODEx: Execute remote request.
  if (!response.ok) { // CODEx: Validate HTTP response.
    throw new Error(`Remote embedding HTTP ${response.status}`); // CODEx: Propagate failure upstream.
  } // CODEx
  ensureJsonResponse(response); // CODEx: Validate JSON body.
  const data = await response.json(); // CODEx: Parse remote payload.
  const vector = normalizeVectorCandidate(data?.data?.[0]?.embedding || data?.embedding); // CODEx: Extract embedding vector.
  if (!vector) { // CODEx: Handle missing embeddings.
    throw new Error('Remote embedding missing'); // CODEx: Raise extraction error.
  } // CODEx
  return vector; // CODEx: Return normalized vector.
} // CODEx
