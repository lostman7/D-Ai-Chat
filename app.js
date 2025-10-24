import { embedAndStore, vectorSearch } from './chunker.js'; // CODEx: Integrate shared embedding utilities for chunk storage and retrieval.
import { EMBEDDING_PROVIDERS, detectEmbeddingService, requestEmbeddingVector } from './rag/local-embed.js'; // CODEx: Incorporate provider-aware embedding helpers.
// CODEx: Maintain explicit separation between module imports and runtime constants.
const MB = 1024 * 1024; // CODEx: Base multiplier for megabyte calculations across memory limits.
const MAX_RAG_EMBEDDING_BYTES = 100 * MB; // CODEx: Cap chunk embedding footprint to 100 MB within the RAM disk budget.
const EMBEDDING_CACHE_MAX_BYTES = MAX_RAG_EMBEDDING_BYTES; // CODEx: Align disk cache limit with RAM disk budget.
const EMBEDDING_CACHE_TRIM_THRESHOLD = Math.floor(EMBEDDING_CACHE_MAX_BYTES * 0.8); // CODEx: Trigger LRU eviction at 80% usage.
const EMBEDDING_BUCKET_KEYS = Object.freeze({ // CODEx: Namespace embedding caches for shared and arena agents.
  SHARED: 'shared', // CODEx: Default bucket for general chat retrieval.
  SAM_A: 'sam-a', // CODEx: Dedicated cache bucket for SAM-A embeddings.
  SAM_B: 'sam-b' // CODEx: Dedicated cache bucket for SAM-B embeddings.
});
const EMBEDDING_CACHE_DIRECTORIES = Object.freeze({ // CODEx: Map cache buckets to their on-disk directories.
  [EMBEDDING_BUCKET_KEYS.SHARED]: 'rag/cache', // CODEx: Shared cache directory.
  [EMBEDDING_BUCKET_KEYS.SAM_A]: 'rag/a_cache', // CODEx: Arena SAM-A cache directory.
  [EMBEDDING_BUCKET_KEYS.SAM_B]: 'rag/b_cache' // CODEx: Arena SAM-B cache directory.
});
const MIN_MEMORY_LIMIT_MB = 50;
const MAX_MEMORY_LIMIT_MB = 200;
const MEMORY_LIMIT_STEP_MB = 10;

const DEFAULT_DUAL_SEED = 'Debate whether persistent memories make AI more helpful.';
// CODEx: Embedding defaults used for vector-based retrieval fallbacks.
const EMBEDDING_DEFAULT_DIMENSIONS = 1536;
const EMBEDDING_HASH_BUCKETS = 512;
const EMBEDDING_MIN_QUERY_LENGTH = 12;

const MODE_CHOOSER = 'chooser';
const MODE_CHAT = 'chat';
const MODE_ARENA = 'arena';

const RAG_STORE_NAME = 'rag-logs';
const RAG_SESSION_KEY_PREFIX = 'sam-rag-session';
const MAX_CHAT_RAG_MESSAGES = 400;
const MAX_ARENA_RAG_MESSAGES = 200;
const ARENA_AUTOSAVE_INTERVAL = 2 * 60 * 1000;
const RAG_CHECKPOINT_SIZE = 30;
const LOG_STORE_NAME = 'sam-logs';
const LOG_STATUS_TIMEOUT = 4500;
const LOG_EXPORT_PREFIX = 'sam-log';
const RETRIEVAL_STORAGE_KEY = 'sam-retrieval-metrics';
const REFLEX_HISTORY_STORE = 'reflex-history';
const DEFAULT_CONTEXT_LIMIT = 131072;
const MAX_CONTEXT_LIMIT = 262144;
const RAG_ORIGIN_TOKENS = ['rag', 'archive', 'checkpoint', 'long-term'];

const STATIC_RAG_MANIFESTS = [
  {
    url: 'rag/manifest.json',
    basePath: 'rag/',
    defaultMode: MODE_CHAT,
    label: 'primary RAG files',
    ramKey: 'rag-manifest'
  },
  {
    url: 'rag/archives/manifest.json',
    basePath: 'rag/archives/',
    defaultMode: MODE_ARENA,
    label: 'RAG archives',
    ramKey: 'rag-archives-manifest'
  }
];
const SUPPORTED_RAG_TEXT_FORMATS = new Set(['txt', 'text', 'md', 'markdown']);
const SUPPORTED_RAG_JSON_FORMATS = new Set(['json', 'jsonl']);
const SUPPORTED_RAG_BINARY_FORMATS = new Set(['pdf']);
const SUPPORTED_CHUNK_EXTENSIONS = new Set([
  ...SUPPORTED_RAG_TEXT_FORMATS,
  ...SUPPORTED_RAG_JSON_FORMATS,
  ...SUPPORTED_RAG_BINARY_FORMATS
]);
const SUPPORTED_CHUNK_CONTENT_TYPES = [
  'text/plain',
  'text/markdown',
  'text/x-markdown',
  'application/json',
  'application/ld+json',
  'application/jsonlines',
  'application/x-ndjson'
];
const PDFJS_CDN_BASE = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.2.67';

const storage = createPersistentStorage();

function createPersistentStorage() {
  if (typeof window !== 'undefined' && window.standaloneStore) {
    const bridge = window.standaloneStore;
    return {
      getItem(key) {
        return bridge.getItem(key);
      },
      setItem(key, value) {
        bridge.setItem(key, value);
      },
      removeItem(key) {
        bridge.removeItem(key);
      },
      keys() {
        try {
          const result = bridge.keys();
          return Array.isArray(result) ? result : [];
        } catch (error) {
          console.warn('Failed to list standalone storage keys:', error);
          return [];
        }
      },
      clear() {
        bridge.clear();
      }
    };
  }

  if (typeof window !== 'undefined' && window.localStorage) {
    return {
      getItem(key) {
        return window.localStorage.getItem(key);
      },
      setItem(key, value) {
        window.localStorage.setItem(key, value);
      },
      removeItem(key) {
        window.localStorage.removeItem(key);
      },
      keys() {
        return Object.keys(window.localStorage);
      },
      clear() {
        window.localStorage.clear();
      }
    };
  }

  const memory = new Map();
  return {
    getItem(key) {
      return memory.has(key) ? memory.get(key) : null;
    },
    setItem(key, value) {
      memory.set(key, String(value));
    },
    removeItem(key) {
      memory.delete(key);
    },
    keys() {
      return Array.from(memory.keys());
    },
    clear() {
      memory.clear();
    }
  };
}

function storageGetItem(key) {
  try {
    return storage.getItem(key);
  } catch (error) {
    console.warn('Failed to read storage key', key, error);
    return null;
  }
}

function storageSetItem(key, value) {
  try {
    storage.setItem(key, String(value));
  } catch (error) {
    console.warn('Failed to persist storage key', key, error);
  }
}

function storageRemoveItem(key) {
  try {
    storage.removeItem(key);
  } catch (error) {
    console.warn('Failed to remove storage key', key, error);
  }
}

function syncReasoningTimeoutToShell() {
  if (typeof window === 'undefined') {
    return; // CODEx: Skip when running outside the browser shell.
  }
  const updater = window.standaloneRuntime?.updateReasoningTimeout;
  if (typeof updater !== 'function') {
    return; // CODEx: Standalone shell not active.
  }
  try {
    updater(config.reasoningTimeoutSeconds); // CODEx: Relay current reasoning timeout to Electron main process.
  } catch (error) {
    console.debug('Failed to sync reasoning timeout to shell', error); // CODEx: Swallow bridge errors without disruption.
  }
}

function storageKeys() {
  try {
    const keys = storage.keys?.();
    if (Array.isArray(keys)) {
      return keys;
    }
    if (typeof storage.length === 'number' && typeof storage.key === 'function') {
      const result = [];
      for (let index = 0; index < storage.length; index += 1) {
        const key = storage.key(index);
        if (key) {
          result.push(key);
        }
      }
      return result;
    }
    return [];
  } catch (error) {
    console.warn('Failed to enumerate storage keys:', error);
    return [];
  }
}

function storageClear() {
  try {
    storage.clear();
  } catch (error) {
    console.warn('Failed to clear storage:', error);
  }
}

function wait(ms) { // CODEx: Promise-based delay helper for retry backoff sequencing.
  return new Promise((resolve) => setTimeout(resolve, Math.max(0, Number(ms) || 0))); // CODEx: Clamp negatives.
}

function computeBackoffDelay(attempt, base = 400, cap = 5000) { // CODEx: Exponential backoff with an upper cap.
  const order = Math.max(0, (Number(attempt) || 1) - 1); // CODEx: Normalize attempt index starting at zero.
  const interval = Math.round((Number(base) || 400) * 2 ** order); // CODEx: Exponentially grow delays per attempt.
  return Math.min(Math.max(0, interval), Number(cap) || 5000); // CODEx: Enforce non-negative durations within cap.
}

function shouldRetryStatus(status) { // CODEx: Identify HTTP statuses that merit automatic retry.
  if (!Number.isFinite(status)) return false; // CODEx: Ignore non-numeric statuses.
  if (status === 429) return true; // CODEx: Back off on rate limits.
  return status >= 500 && status < 600; // CODEx: Retry server-side failures.
}

function isTransientNetworkError(error) { // CODEx: Detect network layer issues eligible for retry.
  if (!error) return false; // CODEx: Guard nullish references.
  if (error.name === 'AbortError') return false; // CODEx: Abort errors handled separately.
  const message = String(error?.message || error || '').toLowerCase(); // CODEx: Normalize message for heuristics.
  if (message.includes('network') || message.includes('fetch') || message.includes('failed to fetch')) {
    return true; // CODEx: Treat generic fetch/network failures as transient.
  }
  return error instanceof TypeError; // CODEx: Browser fetch errors surface as TypeError and should be retried.
}

function getDirectoryPath(path) {
  if (typeof path !== 'string') {
    return '';
  }
  const sanitized = path.split(/[?#]/)[0];
  const lastSlash = sanitized.lastIndexOf('/');
  return lastSlash > 0 ? sanitized.slice(0, lastSlash) : '';
}

function getFileExtension(path) {
  if (typeof path !== 'string') {
    return '';
  }
  const sanitized = path.split(/[?#]/)[0];
  const lastSlash = sanitized.lastIndexOf('/');
  const fileName = lastSlash >= 0 ? sanitized.slice(lastSlash + 1) : sanitized;
  const dotIndex = fileName.lastIndexOf('.');
  if (dotIndex <= 0 || dotIndex === fileName.length - 1) {
    return '';
  }
  return fileName.slice(dotIndex + 1).toLowerCase();
}

function inferContentTypeFromPath(path) {
  const extension = getFileExtension(path);
  if (SUPPORTED_RAG_JSON_FORMATS.has(extension)) {
    return 'application/json';
  }
  if (SUPPORTED_RAG_BINARY_FORMATS.has(extension)) {
    return 'application/pdf';
  }
  return 'text/plain';
}

function getStandaloneFs() {
  if (typeof window === 'undefined') {
    return null;
  }
  return window.standaloneFs ?? null;
}

function ensureStandaloneDirectory(path) {
  const bridge = getStandaloneFs();
  if (!bridge?.ensureDir) return false;
  try {
    return bridge.ensureDir(path) !== false;
  } catch (error) {
    console.warn('Failed to ensure standalone directory', path, error);
    return false;
  }
}

function listStandaloneDirectory(path) {
  const bridge = getStandaloneFs();
  if (!bridge?.listDirectory) return null;
  try {
    const entries = bridge.listDirectory(path);
    return Array.isArray(entries) ? entries : null;
  } catch (error) {
    console.warn('Failed to list standalone directory', path, error);
    return null;
  }
}

function readStandaloneFile(path) {
  const bridge = getStandaloneFs();
  if (!bridge?.readFile) return null;
  try {
    const result = bridge.readFile(path);
    if (!result) return null;
    if (typeof result === 'string') {
      return { content: result, contentType: inferContentTypeFromPath(path) };
    }
    if (typeof result === 'object' && typeof result.content === 'string') {
      return {
        content: result.content,
        contentType: result.contentType || inferContentTypeFromPath(path)
      };
    }
  } catch (error) {
    console.warn('Failed to read standalone file', path, error);
  }
  return null;
}

function writeStandaloneFile(path, content, options = {}) {
  const bridge = getStandaloneFs();
  if (!bridge?.writeFile) return false;
  try {
    const payload = typeof content === 'string' ? content : String(content);
    return bridge.writeFile(path, payload, options) !== false;
  } catch (error) {
    console.warn('Failed to write standalone file', path, error);
    return false;
  }
}

function deleteStandaloneFile(path) {
  const bridge = getStandaloneFs();
  if (!bridge?.deleteFile) return false;
  try {
    return bridge.deleteFile(path) !== false;
  } catch (error) {
    console.warn('Failed to delete standalone file', path, error);
    return false;
  }
}

function isSupportedChunkExtension(extension) {
  if (!extension) return false;
  return SUPPORTED_CHUNK_EXTENSIONS.has(extension.toLowerCase());
}

function isSupportedChunkContentType(contentType) {
  if (!contentType) return false;
  const normalized = contentType.split(';')[0].trim().toLowerCase();
  return SUPPORTED_CHUNK_CONTENT_TYPES.includes(normalized);
}

function isSupportedChunkablePath(path) {
  return isSupportedChunkExtension(getFileExtension(path));
}

function isSupportedChunkableFile(path, contentType) {
  if (isSupportedChunkablePath(path)) {
    return true;
  }
  if (contentType && isSupportedChunkContentType(contentType)) {
    return true;
  }
  return false;
}

function filterChunkablePaths(paths) {
  if (!Array.isArray(paths)) return [];
  return paths.filter((path) => isSupportedChunkablePath(path));
}

const DEFAULT_BACKGROUND =
  'radial-gradient(circle at top, rgba(77, 124, 255, 0.15), transparent 55%), ' +
  'radial-gradient(circle at bottom, rgba(60, 64, 198, 0.1), transparent 45%), ' +
  'linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(236, 238, 245, 0.9))';
const CUSTOM_BACKGROUND_OVERLAY =
  'linear-gradient(135deg, rgba(12, 16, 24, 0.52), rgba(28, 32, 46, 0.42))';

const defaultConfig = {
  memoryLimitMB: 100,
  providerPreset: 'custom',
  retrievalCount: 0,
  reasoningModeEnabled: false,
  autoInjectMemories: false,
  useEmbeddingRetrieval: true, // CODEx: Enable embedding-first retrieval unless explicitly disabled.
  requestTimeoutSeconds: 30,
  reasoningTimeoutSeconds: 300,
  contextTurns: 12,
  endpoint: 'http://localhost:1234/v1/chat/completions',
  model: 'lmstudio-community/Meta-Llama-3-8B-Instruct',
  apiKey: '',
  openRouterPolicy: '',
  systemPrompt: 'You are SAM, a helpful memory-augmented assistant who reflects on long-term memories when they are relevant.',
  temperature: 0.7,
  maxResponseTokens: 4500,
  agentATemperature: 0.7, // CODEx: Agent A specific temperature default.
  agentAMaxResponseTokens: 4500, // CODEx: Agent A specific max tokens default.
  agentBTemperature: 0.7, // CODEx: Agent B specific temperature default.
  agentBMaxResponseTokens: 4500, // CODEx: Agent B specific max tokens default.
  embeddingModel: 'text-embedding-3-large',
  embeddingEndpoint: '',
  embeddingApiKey: '',
  embeddingProviderPreference: EMBEDDING_PROVIDERS.AUTO, // CODEx: Default to automatic provider detection.
  embeddingContextLength: 1024, // CODEx: Default embedding context window.
  autoSpeak: false,
  ttsPreset: 'browser',
  ttsServerUrl: '',
  ttsVoiceId: '',
  ttsApiKey: '',
  ttsVolume: 100,
  agentAName: 'SAM-A',
  agentAPrompt:
    'You are SAM-A, the Flowfield advocate. Defend the Flowfield hypothesis with technical precision, cite mechanisms, and ground arguments in empirical detail.', // CODEx: Flowfield persona for SAM-A.
  agentBEnabled: false,
  agentBName: 'SAM-B',
  agentBPrompt:
    'You are SAM-B, the analytical skeptic. Probe Flowfield assumptions, raise counterexamples, and demand rigorous justification before conceding ground.', // CODEx: Skeptical persona for SAM-B.
  agentBProviderPreset: 'inherit',
  agentBEndpoint: '',
  agentBModel: '',
  agentAEmbeddingProvider: EMBEDDING_PROVIDERS.OLLAMA_LOCAL, // CODEx: Default SAM-A embedding provider.
  agentAEmbeddingModel: 'mxbai-embed-large', // CODEx: Default SAM-A embedding checkpoint.
  agentAEmbeddingEndpoint: '', // CODEx: SAM-A specific embedding endpoint override.
  agentAEmbeddingApiKey: '', // CODEx: SAM-A specific embedding API key.
  agentAEmbeddingContextLength: 1024, // CODEx: SAM-A embedding context hint.
  agentBEmbeddingProvider: EMBEDDING_PROVIDERS.LM_STUDIO, // CODEx: Default SAM-B embedding provider.
  agentBEmbeddingModel: 'text-embedding-3-large', // CODEx: Default SAM-B embedding checkpoint.
  agentBEmbeddingEndpoint: '', // CODEx: SAM-B specific embedding endpoint override.
  agentBEmbeddingApiKey: '', // CODEx: SAM-B specific embedding API key.
  agentBEmbeddingContextLength: 1024, // CODEx: SAM-B embedding context hint.
  agentBApiKey: '',
  dualAutoContinue: true,
  dualTurnLimit: 0,
  dualTurnDelaySeconds: 15,
  dualSeed: DEFAULT_DUAL_SEED,
  reflexEnabled: true,
  reflexInterval: 8,
  entropyWindow: 5,
  backgroundSource: 'default',
  backgroundImage: '',
  backgroundUrl: '',
  debugEnabled: false
};

const phases = [
  { id: 1, title: 'Flow Dynamics', prompt: 'phase1.txt' },
  { id: 2, title: 'Collapse Engine', prompt: 'phase2.txt' },
  { id: 3, title: 'Entanglement Node', prompt: 'phase3.txt' }
];

const PHASE_PROMPT_DIR = 'rag/phase-prompts/';
const WATCHDOG_MESSAGE_SOURCE = 'sam-standalone-watchdog';
const WATCHDOG_MESSAGE_TYPE = 'check-arena-memory';
const WATCHDOG_PRESSURE_THRESHOLD = 0.8;
const WATCHDOG_ARCHIVE_PERCENT = 0.1;
const ENTROPY_LOOP_THRESHOLD = 0.8;
const ENTROPY_EXPLORE_THRESHOLD = 0.3;
const LOOP_DELAY_SCALE = 0.6;
const EXPLORE_DELAY_SCALE = 1.4;
const MIN_DYNAMIC_DELAY_SECONDS = 5;
const MAX_DYNAMIC_DELAY_SECONDS = 120;

const TEST_TTS_PHRASE = 'This is SAM performing a voice check with persistent memory engaged.';
const TTS_ERROR_COOLDOWN = 15000;
const PINNED_STORAGE_KEY = 'sam-pinned-messages';
const RAM_DISK_CHUNK_PREFIX = 'sam-chunked-';
const RAM_DISK_UNCHUNKED_PREFIX = 'sam-unchunked-';
const RAM_DISK_ARCHIVE_PREFIX = 'sam-archive-';
const RAM_DISK_MANIFEST_PREFIX = 'sam-manifest-';
const REASONING_MODEL_HINTS = [
  'reason', // CODEx: Preserve legacy hints for bespoke preset labels.
  'cogito', // CODEx: Retain prior identifier coverage for specialized checkpoints.
  'sonar', // CODEx: Keep community alias alignment.
  'think', // CODEx: Maintain compatibility with earlier think-prefixed models.
  'deepseek-r1', // CODEx: Support DeepSeek reasoning suite detection.
  'deepseek_reasoner', // CODEx: Alias for DeepSeek reasoning models.
  'reasoner', // CODEx: Capture general reasoner naming conventions.
  'o1', // CODEx: Include OpenAI o-series heuristics.
  'o3', // CODEx: Preserve existing o-series mapping.
  'reflect', // CODEx: Extend hints for reflect-oriented reasoning checkpoints.
  'cot' // CODEx: Support chain-of-thought model aliases.
];
const REASONING_MODEL_REGEX = /reason|think|deep|reflect|cot|chain|reflection/i; // CODEx: Phase VI expanded reasoning detector.
const EMBEDDING_MODEL_REGEX = /embed|embedding|text[-_]?embedding|e5|gte-|nv-embed|textembedding/i; // CODEx: Identify embedding-specialized checkpoints including modern aliases.
const CHAT_ROUTES = Object.freeze({ // CODEx: Centralize provider-specific chat endpoints.
  'openai-like': '/v1/chat/completions', // CODEx: Default OpenAI-compatible route.
  'lmstudio-local': '/v1/chat/completions', // CODEx: LM Studio aligns with OpenAI schema for chat.
  'ollama-local': '/api/generate', // CODEx: Native Ollama chat endpoint.
  'ollama-cloud': '/v1/chat/completions', // CODEx: Ollama Cloud mirrors OpenAI routes.
  openrouter: '/v1/chat/completions' // CODEx: OpenRouter leverages OpenAI-compatible schema.
});
const EMBED_ROUTES = Object.freeze({ // CODEx: Centralize provider-specific embedding endpoints.
  'openai-like': '/v1/embeddings', // CODEx: Default embeddings route for OpenAI-compatible APIs.
  'lmstudio-local': '/v1/embeddings', // CODEx: LM Studio local embeddings endpoint.
  'ollama-local': '/api/embed', // CODEx: Native Ollama embedding endpoint.
  'ollama-cloud': '/v1/embeddings', // CODEx: Ollama Cloud exposes OpenAI-compatible embeddings route.
  openrouter: '/v1/embeddings' // CODEx: OpenRouter embeddings mirror OpenAI routes.
});

function getProviderRouteKey(providerId) { // CODEx: Map provider identifiers to standardized routing keys.
  const id = (providerId || '').toString().toLowerCase(); // CODEx: Normalize provider identifier for comparisons.
  if (id === EMBEDDING_PROVIDERS.LM_STUDIO || id === 'lmstudio' || id === 'lmstudio-local') {
    return 'lmstudio-local';
  }
  if (id === EMBEDDING_PROVIDERS.OLLAMA_LOCAL || id === 'ollama' || id === 'ollama-local') {
    return 'ollama-local';
  }
  if (id === EMBEDDING_PROVIDERS.OLLAMA_CLOUD || id === 'ollama-cloud') {
    return 'ollama-cloud';
  }
  if (id === 'openrouter') {
    return 'openrouter';
  }
  return 'openai-like'; // CODEx: Default to OpenAI-compatible routing.
}

function joinProviderEndpoint(baseUrl, suffix) { // CODEx: Combine a base URL with a provider route suffix.
  const base = stripTrailingSlashes(baseUrl || ''); // CODEx: Ensure base lacks trailing slash before concatenation.
  if (!base) {
    return suffix || '';
  }
  const cleanedSuffix = (suffix || '').replace(/^\/+/, ''); // CODEx: Avoid duplicate slashes when joining.
  return `${base}/${cleanedSuffix}`;
}

function deriveBaseOrigin(url) { // CODEx: Reduce a URL to its scheme and host for provider routing.
  if (!url) {
    return '';
  }
  try {
    const parsed = new URL(url, typeof window !== 'undefined' ? window.location.origin : undefined); // CODEx: Resolve relative URLs when running in browser.
    return `${parsed.protocol}//${parsed.host}`.replace(/\/$/, ''); // CODEx: Return scheme + host without trailing slash.
  } catch (error) {
    const sanitized = String(url).split(/[?#]/)[0]; // CODEx: Drop query/hash fragments manually.
    return sanitized.replace(/\/[^/]*$/, '').replace(/\/$/, ''); // CODEx: Remove trailing path components and slash.
  }
}

const providerPresets = [
  {
    id: 'custom',
    label: 'Custom / bring your own',
    description:
      'Wire up any OpenAI-compatible API by entering your own endpoint, model ID, and optional headers. Perfect for bespoke gateways or experimental backends.',
    endpoint: '',
    model: '',
    requiresKey: false,
    contextLimit: 131072
  },
  {
    id: 'lmstudio',
    label: 'LM Studio (local server)',
    description:
      'Use LM Studio\'s local server mode. Start LM Studio, enable "Local Server" in the sidebar, and keep the default port 1234.',
    endpoint: 'http://localhost:1234/v1/chat/completions',
    model: 'lmstudio-community/TinyLlama-1.1B-Chat-v1.0',
    requiresKey: false,
    contextLimit: 131072
  },
  {
    id: 'ollama',
    label: 'Ollama (local)',
    description:
      'Point SAM at an Ollama instance. Run `ollama serve`, pull a model such as `tinyllama`, and enable the OpenAI-compatible endpoint.',
    endpoint: 'http://localhost:11434/api/generate', // CODEx hot-fix: align Ollama local default with native generate route.
    model: 'tinyllama',
    requiresKey: false,
    contextLimit: 65536
  },
  {
    id: 'ollama-cloud', // CODEx: Expose Ollama Cloud as a provider preset.
    label: 'Ollama Cloud', // CODEx: UI label for Ollama Cloud connections.
    description:
      'Route SAM through Ollama Cloud. Provide your OLLAMA_API_KEY and the service will mirror OpenAI-compatible chat completions.', // CODEx: Explain Ollama Cloud usage.
    endpoint: 'https://api.ollama.ai/v1/chat/completions', // CODEx: Default Ollama Cloud chat endpoint.
    model: 'llama3.1', // CODEx: Reasonable default model identifier for cloud usage.
    requiresKey: true, // CODEx: Ollama Cloud requires an API key.
    contextLimit: 65536 // CODEx: Estimated context limit for Ollama Cloud.
  },
  {
    id: 'openrouter',
    label: 'OpenRouter (cloud hub)',
    description:
      'Connect via OpenRouter to try community-ranked models. Add your API key and optional Referer/Title headers in OpenRouter settings.',
    endpoint: 'https://openrouter.ai/api/v1/chat/completions',
    model: 'openrouter/google/gemma-2-9b-it',
    requiresKey: true,
    contextLimit: 128000
  },
  {
    id: 'openai',
    label: 'OpenAI',
    description:
      'Use OpenAI\'s latest GPT endpoints. Requires an API key with billing enabled.',
    endpoint: 'https://api.openai.com/v1/chat/completions',
    model: 'gpt-4o',
    requiresKey: true,
    contextLimit: 128000
  },
  {
    id: 'groq',
    label: 'Groq (LPU cloud)',
    description:
      'Tap into Groq\'s low-latency hosting for Llama 3 and Mixtral families. Provide your Groq API key and choose a supported model.',
    endpoint: 'https://api.groq.com/openai/v1/chat/completions',
    model: 'llama-3.2-90b-text-preview',
    requiresKey: true,
    contextLimit: 32768
  },
  {
    id: 'together',
    label: 'Together AI',
    description:
      'Route through Together AI for curated open-source checkpoints. Requires an API key from dashboard.together.ai.',
    endpoint: 'https://api.together.xyz/v1/chat/completions',
    model: 'meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo',
    requiresKey: true,
    contextLimit: 131072
  },
  {
    id: 'mistral',
    label: 'Mistral',
    description:
      'Access Mistral\'s hosted models via their OpenAI-compatible endpoint. Supply your Mistral API key.',
    endpoint: 'https://api.mistral.ai/v1/chat/completions',
    model: 'mistral-large-2411',
    requiresKey: true,
    contextLimit: 128000
  },
  {
    id: 'perplexity',
    label: 'Perplexity',
    description:
      'Connect to Perplexity\'s reasoning-oriented models. Requires a Perplexity API key with chat access.',
    endpoint: 'https://api.perplexity.ai/chat/completions',
    model: 'llama-3.1-sonar-large-128k-online',
    requiresKey: true,
    contextLimit: 128000
  },
  {
    id: 'fireworks',
    label: 'Fireworks AI',
    description:
      'Use Fireworks AI for high-context Llama and Mixtral variants. Provide your Fireworks API key.',
    endpoint: 'https://api.fireworks.ai/inference/v1/chat/completions',
    model: 'accounts/fireworks/models/llama-v3p2-90b-vision-instruct',
    requiresKey: true,
    contextLimit: 128000
  },
  {
    id: 'deepseek',
    label: 'DeepSeek',
    description:
      'Route to DeepSeek\'s hosted reasoning models. Requires a DeepSeek API key and matching model name.',
    endpoint: 'https://api.deepseek.com/v1/chat/completions',
    model: 'deepseek-reasoner',
    requiresKey: true,
    contextLimit: 64000
  },
  {
    id: 'xai',
    label: 'xAI (Grok)',
    description:
      'Access xAI\'s Grok models directly. Requires an xAI API key from x.ai.',
    endpoint: 'https://api.x.ai/v1/chat/completions',
    model: 'grok-2-1212',
    requiresKey: true,
    contextLimit: 131072
  },
  {
    id: 'anthropic',
    label: 'Anthropic (Claude)',
    description:
      'Use Anthropic\'s Claude models via their official API. Requires an Anthropic API key.',
    endpoint: 'https://api.anthropic.com/v1/messages',
    model: 'claude-3-5-sonnet-20241022',
    requiresKey: true,
    contextLimit: 200000,
    anthropicFormat: true
  },
  {
    id: 'google',
    label: 'Google (Gemini)',
    description:
      'Access Google\'s Gemini models through their AI Studio API. Requires a Google AI API key.',
    endpoint: 'https://generativelanguage.googleapis.com/v1beta/models/',
    model: 'gemini-1.5-pro',
    requiresKey: true,
    contextLimit: 2097152,
    googleFormat: true
  }
];

const providerPresetMap = new Map(providerPresets.map((preset) => [preset.id, preset]));

const ttsPresets = [
  {
    id: 'browser',
    label: 'Browser voice (Web Speech API)',
    provider: 'browser',
    rating: '⭐️ 3.6 avg.',
    hardware: 'Any CPU',
    languages: 'Depends on installed system voices',
    description: 'Zero-setup fallback that speaks with whatever voices your operating system provides.',
    notes: 'Quality varies per platform. Ideal as a backup when remote engines are offline.',
    voiceHint: 'Choose from the browser voice catalog below once loaded.'
  },
  {
    id: 'piper-amy',
    label: 'Piper • en_US-amy-low',
    provider: 'piper',
    rating: '⭐️ 4.7 (Mycroft community)',
    hardware: 'AMD CPU / GPU via ONNX Runtime ROCm',
    languages: 'English',
    description: 'Fast neural TTS tuned for offline setups and AMD-friendly via onnxruntime-rocm.',
    notes: 'Start the Piper HTTP server with `piper-tts --http 0.0.0.0:5002 --voice en_US-amy-low`.',
    defaultUrl: 'http://localhost:5002',
    endpoint: '/synthesize',
    voiceId: 'en_US-amy-low',
    sampleRate: 22050,
    serverHint: 'http://localhost:5002',
    voiceHint: 'e.g. en_US-amy-low or other Piper voice IDs'
  },
  {
    id: 'coqui-xtts-v2',
    label: 'Coqui XTTS v2',
    provider: 'coqui_xtts',
    rating: '⭐️ 4.8 (Coqui community)',
    hardware: 'AMD GPU/CPU',
    languages: 'Multilingual',
    description: 'Cross-lingual, expressive voices with quick fine-tuning support and AMD acceleration via DirectML/ROCm.',
    notes: 'Run `tts --model XTTS-v2 --port 8021 --use_cuda 0` and supply optional reference audio for cloning.',
    defaultUrl: 'http://localhost:8021',
    endpoint: '/api/tts',
    voiceId: 'female-en',
    language: 'en',
    voiceHint: 'Coqui speaker name, e.g. female-en or your custom voice ID',
    serverHint: 'http://localhost:8021'
  },
  {
    id: 'bark-small',
    label: 'Suno Bark-small (AMD build)',
    provider: 'bark',
    rating: '⭐️ 4.4 (community packs)',
    hardware: 'AMD GPU (ROCm) or CPU',
    languages: 'Multilingual',
    description: 'Expressive speech with emotions and music-like prosody. Works with AMD-friendly Bark HTTP servers.',
    notes: 'Use a Bark web server such as `bark-server --host 0.0.0.0 --port 5005` with patched AMD wheels.',
    defaultUrl: 'http://localhost:5005',
    endpoint: '/generate',
    voiceId: 'en_speaker_6',
    voiceHint: 'Bark speaker token, e.g. en_speaker_6',
    serverHint: 'http://localhost:5005'
  },
  {
    id: 'mimic3',
    label: 'Mycroft Mimic 3',
    provider: 'mimic3',
    rating: '⭐️ 4.6 (Mycroft community)',
    hardware: 'AMD CPU / GPU via onnxruntime-rocm',
    languages: 'English + select locales',
    description: 'Open-source neural TTS with curated Mycroft voices and pronunciation dictionary support.',
    notes: 'Launch with `mimic3-server --host 0.0.0.0 --port 59125 --voice en_US/mimic3_low`.',
    defaultUrl: 'http://localhost:59125',
    endpoint: '/api/tts',
    voiceId: 'en_US/mimic3_low',
    voiceHint: 'Installed Mimic 3 voice, e.g. en_US/mimic3_low',
    serverHint: 'http://localhost:59125'
  },
  {
    id: 'f5-tts',
    label: 'F5-TTS (Glow/FastPitch hybrid)',
    provider: 'f5',
    rating: '⭐️ 4.5 (community demos)',
    hardware: 'AMD GPU (ROCm) / CPU',
    languages: 'English',
    description: 'Fast expressive speech with easy fine-tuning; great for lightweight AMD rigs.',
    notes: 'Serve with `f5-tts-server --host 0.0.0.0 --port 8088` or the TTS WebUI plugin.',
    defaultUrl: 'http://localhost:8088',
    endpoint: '/api/tts',
    voiceId: 'f5_tts_default',
    voiceHint: 'F5 speaker or checkpoint ID, e.g. f5_tts_default',
    serverHint: 'http://localhost:8088'
  },
  {
    id: 'elevenlabs',
    label: 'ElevenLabs (cloud)',
    provider: 'elevenlabs',
    rating: '⭐️ 4.9 (user reviews)',
    hardware: 'Cloud-hosted',
    languages: 'English + multilingual (select voices)',
    description: 'Top-rated commercial voice quality with instant results; requires an API key.',
    notes: 'Set your voice ID in the field below and provide the ElevenLabs API key.',
    defaultUrl: 'https://api.elevenlabs.io/v1/text-to-speech',
    endpoint: '',
    needsApiKey: true,
    modelId: 'eleven_monolingual_v1',
    voiceHint: 'ElevenLabs voice ID, e.g. 21m00Tcm4TlvDq8ikWAM',
    serverHint: 'https://api.elevenlabs.io/v1/text-to-speech'
  }
];

const ttsPresetMap = new Map(ttsPresets.map((preset) => [preset.id, preset]));

let config = { ...defaultConfig };

let db;
let floatingMemory = [];
let conversationLog = [];
let pinnedMessageIds = new Set();
let recognition;
let isVoiceActive = false;
let isDualChatRunning = false;
let nextDualSpeaker = 'A';
let dualChatHistory = [];
let dualTurnsCompleted = 0;
let dualTurnCounter = 0;
const activeDualConnections = {
  A: null,
  B: null
};
let autoContinueTimer;
let dualAutosaveTimer;
let dualCountdownTimer;
let dualCountdownDeadline = null;
let audioContext;
let lastTtsErrorAt = 0;
let userEditedTtsServer = false;
let userEditedTtsVoice = false;
let turnCounter = 0;
let activeDualSeed = DEFAULT_DUAL_SEED;
let activeMode = MODE_CHOOSER;
let activeChatSessionId;
let activeArenaSessionId;
let chatCheckpointBuffer = [];
let arenaCheckpointBuffer = [];
let reflexInFlight = false;
let lastReflexSummaryAt = null;
let reflexSummaryCount = 0;
let currentEntropyScore = 0;
let lastEntropyState = 'calibrating';
let deviceMemoryEstimate;
let gpuRendererInfo;
let logEntries = [];
let logStatusTimer;
let retrievalMetrics = {
  A: { total: 0, lastCount: 0, lastAt: null },
  B: { total: 0, lastCount: 0, lastAt: null }
};
let ragTelemetry = {
  lastLoad: null,
  lastLoadCount: 0,
  lastLoadRecords: 0,
  lastLoadBytes: 0,
  lastSave: null,
  lastSaveMode: null,
  staticFiles: 0,
  staticBytes: 0,
  staticErrors: [],
  loaded: false,
  lastRetrievalMs: 0,
  lastRetrievalMode: 'lexical' // CODEx: Track whether embeddings or keyword fallback handled the last retrieval.
};
const ramDiskCache = {
  chunked: new Map(),
  unchunked: new Map(),
  archives: new Map(),
  manifests: new Map()
};
// CODEx: In-memory embedding caches maintain fast retrieval across chat and RAG stores.
const embeddingCache = new Map();
const messageEmbeddingCache = new Map();
const embeddingRuntimeBuckets = new Map(); // CODEx: Track in-memory embedding state per cache bucket.
let totalEmbeddingFootprintBytes = 0; // CODEx: Aggregate bytes consumed by all embedding buckets.
let embeddingServiceHealthy = true; // CODEx: Represent whether the embedding endpoint is responsive.
let embeddingServiceReason = ''; // CODEx: Store the most recent embedding failure reason for UI diagnostics.
const embeddingProviderState = { preference: EMBEDDING_PROVIDERS.AUTO, active: EMBEDDING_PROVIDERS.OPENAI, baseUrl: '', lastProbe: 0 }; // CODEx: Track provider selection and probe metadata.
const embeddingBucketManifests = new Map(); // CODEx: Persisted manifest metadata keyed by cache bucket.
const embeddingBucketManifestLoaded = new Set(); // CODEx: Track which manifests have been hydrated from disk.

// CODEx: High-resolution clock helper for latency tracking.
function getTimestampMs() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

// CODEx: Model call metrics inform diagnostics and timeout reporting.
const modelCallMetrics = {
  totalMs: 0,
  count: 0,
  reasoningTimeouts: 0
};

// CODEx: Reset embedding caches when the embedding model or endpoint changes.
function clearCachedEmbeddings() {
  embeddingCache.clear();
  messageEmbeddingCache.clear();
  embeddingRuntimeBuckets.clear(); // CODEx: Reset per-bucket embedding state for shared and arena caches.
  totalEmbeddingFootprintBytes = 0; // CODEx: Reset aggregate embedding footprint across all buckets.
}

// CODEx: Detect reasoning-focused checkpoints to adjust timeouts and context.
function isReasoningModelId(modelId, preset) {
  if (preset?.anthropicFormat) {
    return true; // CODEx: Anthropic chat endpoints are treated as reasoning-safe by default.
  }
  const normalized = typeof modelId === 'string' ? modelId.toLowerCase() : ''; // CODEx: Normalize identifiers for regex tests.
  if (!normalized) {
    return false; // CODEx: Skip empty identifiers.
  }
  if (REASONING_MODEL_REGEX.test(normalized)) {
    return true; // CODEx: Apply Phase II reasoning handshake detector.
  }
  return REASONING_MODEL_HINTS.some((hint) => normalized.includes(hint)); // CODEx: Retain heuristic fallback coverage.
}

// CODEx: Detect embedding-only checkpoints to avoid routing them through text generation endpoints.
function isEmbeddingModelId(modelId) {
  if (typeof modelId !== 'string') {
    return false; // CODEx: Non-string identifiers cannot be matched reliably.
  }
  return EMBEDDING_MODEL_REGEX.test(modelId.toLowerCase()); // CODEx: Apply embedding model heuristic pattern.
}

function ensureChatModel(modelId, providerPreset) { // CODEx: Guard against embedding checkpoints in chat routes.
  if (!isEmbeddingModelId(modelId)) {
    return modelId; // CODEx: Safe to reuse requested model when it supports chat.
  }
  const fallback = providerPreset?.model || defaultConfig.model; // CODEx: Defer to preset defaults when available.
  console.warn('[SAM] Chat requested with embedding model; switching to fallback', { requested: modelId, fallback }); // CODEx: Surface automatic fallback telemetry.
  return fallback || modelId; // CODEx: Preserve original identifier if no safer fallback exists.
}

// CODEx: Determine whether reasoning safeguards should be active for the current request.
function isReasoningModeActive(overrides = {}) {
  if (overrides.reasoningMode === true) {
    return true;
  }
  if (overrides.reasoningMode === false) {
    return false;
  }
  if (config.reasoningModeEnabled) {
    return true;
  }
  const preset = providerPresetMap.get(overrides.providerPreset ?? config.providerPreset);
  const model = overrides.model ?? overrides.modelId ?? config.model;
  return isReasoningModelId(model, preset);
}

// CODEx: Embedding retrieval helpers manage vector health and budgeting.
function isEmbeddingRetrievalEnabled() { // CODEx: Determine if embedding search should run for the current configuration.
  return config.useEmbeddingRetrieval !== false; // CODEx: Default to enabling embeddings unless explicitly disabled.
} // CODEx
// CODEx: Maintain accurate embedding health tracking for UI diagnostics.
function setEmbeddingServiceHealth(healthy, reason = '') { // CODEx: Persist the embedding provider health status.
  embeddingServiceHealthy = Boolean(healthy); // CODEx: Normalize provider health to a strict boolean.
  embeddingServiceReason = healthy ? '' : String(reason || '').slice(0, 60); // CODEx: Retain a trimmed explanation for UI messaging.
  refreshEmbeddingStatusIndicator(); // CODEx: Keep the embedding status indicator synchronized with the latest health state.
} // CODEx
function getActiveEmbeddingProviderLabel() { // CODEx: Present user-friendly provider labels.
  switch (embeddingProviderState.active) { // CODEx: Branch on current provider value.
    case EMBEDDING_PROVIDERS.LM_STUDIO:
      return 'LMStudio'; // CODEx: Normalize LM Studio label casing.
    case EMBEDDING_PROVIDERS.OLLAMA_LOCAL:
      return 'Ollama-Local'; // CODEx: Standardize Ollama local label.
    case EMBEDDING_PROVIDERS.OLLAMA_CLOUD:
      return 'Ollama-Cloud'; // CODEx: Standardize Ollama cloud label.
    case EMBEDDING_PROVIDERS.OPENAI:
      return 'OpenAI'; // CODEx: Remote provider label.
    default:
      return 'Auto'; // CODEx: Fallback descriptor when detection pending.
  }
} // CODEx
function normalizeEmbeddingBucket(bucketKey) { // CODEx: Resolve a valid embedding bucket identifier.
  if (bucketKey && EMBEDDING_CACHE_DIRECTORIES[bucketKey]) { // CODEx: Accept known bucket keys directly.
    return bucketKey; // CODEx: Use provided bucket when valid.
  }
  if (bucketKey === 'A' || bucketKey === 'SAM-A') { // CODEx: Map legacy speaker identifiers.
    return EMBEDDING_BUCKET_KEYS.SAM_A; // CODEx: Normalize to SAM-A bucket.
  }
  if (bucketKey === 'B' || bucketKey === 'SAM-B') { // CODEx: Map legacy speaker identifiers.
    return EMBEDDING_BUCKET_KEYS.SAM_B; // CODEx: Normalize to SAM-B bucket.
  }
  return EMBEDDING_BUCKET_KEYS.SHARED; // CODEx: Default to shared cache for unknown identifiers.
} // CODEx

function getRuntimeEmbeddingBucket(bucketKey = EMBEDDING_BUCKET_KEYS.SHARED) { // CODEx: Access runtime cache state for a bucket.
  const normalized = normalizeEmbeddingBucket(bucketKey); // CODEx: Ensure consistent bucket key usage.
  if (!embeddingRuntimeBuckets.has(normalized)) { // CODEx: Lazy-initialize bucket state.
    embeddingRuntimeBuckets.set(normalized, { index: new Map(), order: [], footprint: 0 }); // CODEx: Seed map, eviction order, and footprint.
  }
  return embeddingRuntimeBuckets.get(normalized); // CODEx: Return cached state container.
} // CODEx

function ensureEmbeddingCacheManifest(bucketKey = EMBEDDING_BUCKET_KEYS.SHARED) { // CODEx: Lazy-load manifest for a bucket.
  const normalized = normalizeEmbeddingBucket(bucketKey); // CODEx: Normalize identifier.
  if (embeddingBucketManifestLoaded.has(normalized)) { // CODEx: Skip reload when already hydrated.
    return; // CODEx: Manifest already ready.
  }
  embeddingBucketManifestLoaded.add(normalized); // CODEx: Mark manifest as loaded (or attempted).
  const directory = EMBEDDING_CACHE_DIRECTORIES[normalized]; // CODEx: Resolve bucket directory.
  const manifestPath = `${directory}/manifest.json`; // CODEx: Derive manifest location for the bucket.
  try { // CODEx: Load manifest content from disk.
    ensureStandaloneDirectory(directory); // CODEx: Create directory when running in standalone shell.
    const record = readStandaloneFile(manifestPath); // CODEx: Read persisted manifest payload.
    if (record?.content) { // CODEx: Parse JSON content when present.
      const parsed = JSON.parse(record.content); // CODEx: Interpret manifest JSON.
      if (parsed && typeof parsed === 'object') { // CODEx: Validate structure before storing.
        embeddingBucketManifests.set(normalized, { // CODEx: Cache manifest entries by bucket.
          entries: Array.isArray(parsed.entries) ? parsed.entries : [], // CODEx: Normalize entry array.
          totalBytes: Number.isFinite(parsed.totalBytes) ? parsed.totalBytes : 0 // CODEx: Normalize footprint.
        });
        return; // CODEx: Manifest loaded successfully.
      }
    }
  } catch (error) { // CODEx: Ignore manifest load failures while logging diagnostics.
    console.warn('Failed to load embedding cache manifest', { bucket: normalized, error }); // CODEx: Surface manifest load issues per bucket.
  }
  embeddingBucketManifests.set(normalized, { entries: [], totalBytes: 0 }); // CODEx: Seed empty manifest on failure.
} // CODEx

function getEmbeddingManifest(bucketKey = EMBEDDING_BUCKET_KEYS.SHARED) { // CODEx: Retrieve hydrated manifest for a bucket.
  const normalized = normalizeEmbeddingBucket(bucketKey); // CODEx: Normalize identifier for lookup.
  ensureEmbeddingCacheManifest(normalized); // CODEx: Guarantee manifest is loaded from disk.
  return embeddingBucketManifests.get(normalized) ?? { entries: [], totalBytes: 0 }; // CODEx: Provide manifest fallback when missing.
} // CODEx

function persistEmbeddingCacheManifest(bucketKey = EMBEDDING_BUCKET_KEYS.SHARED) { // CODEx: Write manifest metadata to disk.
  const normalized = normalizeEmbeddingBucket(bucketKey); // CODEx: Normalize identifier for persistence.
  const manifest = getEmbeddingManifest(normalized); // CODEx: Retrieve manifest snapshot.
  const directory = EMBEDDING_CACHE_DIRECTORIES[normalized]; // CODEx: Bucket directory on disk.
  const manifestPath = `${directory}/manifest.json`; // CODEx: Derived manifest path.
  try { // CODEx: Serialize manifest to JSON.
    ensureStandaloneDirectory(directory); // CODEx: Ensure directory exists prior to writing.
    const payload = JSON.stringify({ entries: manifest.entries, totalBytes: manifest.totalBytes }); // CODEx: Compose manifest payload.
    const saved = writeStandaloneFile(manifestPath, payload); // CODEx: Attempt to write manifest to disk.
    if (saved === false) { // CODEx: Skip when persistence unavailable.
      return; // CODEx: Allow in-memory operation without disk support.
    }
  } catch (error) { // CODEx: Handle persistence issues gracefully.
    console.warn('Failed to persist embedding cache manifest', { bucket: normalized, error }); // CODEx: Surface disk write failures.
  }
} // CODEx

function getEmbeddingCacheBytes(bucketKey) { // CODEx: Report cache footprint per bucket or aggregate.
  if (bucketKey) { // CODEx: Provide targeted bucket totals when requested.
    const manifest = getEmbeddingManifest(bucketKey); // CODEx: Retrieve bucket manifest.
    return Number.isFinite(manifest.totalBytes) ? manifest.totalBytes : 0; // CODEx: Safely return numeric footprint.
  }
  let total = 0; // CODEx: Aggregate totals across buckets.
  for (const key of Object.values(EMBEDDING_BUCKET_KEYS)) { // CODEx: Iterate all known buckets.
    total += getEmbeddingCacheBytes(key); // CODEx: Sum each bucket footprint.
  }
  return total; // CODEx: Provide combined footprint when no bucket specified.
} // CODEx

function formatEmbeddingCacheBreakdown() { // CODEx: Present cache footprint per bucket for UI status.
  const shared = formatMegabytes(getEmbeddingCacheBytes(EMBEDDING_BUCKET_KEYS.SHARED)); // CODEx: Shared cache footprint label.
  const aBytes = formatMegabytes(getEmbeddingCacheBytes(EMBEDDING_BUCKET_KEYS.SAM_A)); // CODEx: SAM-A cache footprint label.
  const bBytes = formatMegabytes(getEmbeddingCacheBytes(EMBEDDING_BUCKET_KEYS.SAM_B)); // CODEx: SAM-B cache footprint label.
  return `Shared ${shared} • SAM-A ${aBytes} • SAM-B ${bBytes}`; // CODEx: Consolidated status string for the drawer.
} // CODEx

function refreshEmbeddingStatusIndicator() { // CODEx: Update the memory drawer indicator for embedding availability.
  const indicator = elements.embeddingStatus; // CODEx: Reference the DOM node used for embedding status text.
  if (!indicator) { // CODEx: Exit early when the indicator has not been rendered.
    return; // CODEx: Avoid errors when the status element is absent.
  } // CODEx
  if (!isEmbeddingRetrievalEnabled()) { // CODEx: Display disabled messaging when embeddings are turned off.
    indicator.textContent = 'Embedding Active ✗ (disabled)'; // CODEx: Communicate the disabled embedding state to the user.
    return; // CODEx: Skip additional status updates when disabled.
  } // CODEx
  const providerLabel = getActiveEmbeddingProviderLabel(); // CODEx: Gather current provider label.
  const cacheLabel = formatEmbeddingCacheBreakdown(); // CODEx: Summarize per-bucket cache footprint for diagnostics.
  if (embeddingServiceHealthy) { // CODEx: Show success when the embedding provider is responding.
    indicator.textContent = `Embedding Active ✓ (Provider: ${providerLabel}, Cache: ${cacheLabel})`; // CODEx: Display provider and cache footprint on success.
    return; // CODEx: Finish once success state is displayed.
  } // CODEx
  const reason = embeddingServiceReason ? ` (${embeddingServiceReason})` : ' (offline)'; // CODEx: Prepare contextual failure messaging.
  indicator.textContent = `Embedding Active ✗ (Provider: ${providerLabel}, Cache: ${cacheLabel})${reason}`; // CODEx: Combine failure message with provider context.
} // CODEx

function updateCardSectionState(cardElement, toggleButton, bodyElement, collapsed) { // CODEx: Apply collapse state to accordion cards.
  if (!cardElement || !toggleButton || !bodyElement) { // CODEx: Guard against missing DOM nodes.
    return; // CODEx: Bail when any element is unavailable.
  }
  const nextCollapsed = Boolean(collapsed); // CODEx: Normalize boolean flag.
  cardElement.setAttribute('data-collapsed', String(nextCollapsed)); // CODEx: Persist collapse attribute for styling.
  toggleButton.textContent = nextCollapsed ? '▸' : '▾'; // CODEx: Reflect collapse state via chevron glyph.
  toggleButton.setAttribute('aria-expanded', String(!nextCollapsed)); // CODEx: Maintain accessibility hint.
  bodyElement.hidden = nextCollapsed; // CODEx: Toggle body visibility while retaining state.
} // CODEx

function toggleCardSection(cardElement, toggleButton, bodyElement) { // CODEx: Flip accordion state on demand.
  if (!cardElement || !toggleButton || !bodyElement) { // CODEx: Skip when card parts are missing.
    return; // CODEx: Avoid errors for incomplete markup.
  }
  const collapsed = cardElement.getAttribute('data-collapsed') === 'true'; // CODEx: Read current collapse flag.
  updateCardSectionState(cardElement, toggleButton, bodyElement, !collapsed); // CODEx: Apply inverted state.
} // CODEx

async function runEmbeddingTest(agent) { // CODEx: Probe embedding health for a specific arena agent.
  const normalizedAgent = agent === 'B' ? 'B' : 'A'; // CODEx: Default to SAM-A when unspecified.
  const statusElement = normalizedAgent === 'A' ? elements.agentAEmbeddingStatus : elements.agentBEmbeddingStatus; // CODEx: Resolve status label target.
  if (statusElement) {
    statusElement.textContent = 'Testing…'; // CODEx: Provide immediate feedback to the user.
  }
  const sampleText = `Embedding diagnostics ping for SAM-${normalizedAgent}`; // CODEx: Deterministic sample phrase.
  const started = getTimestampMs(); // CODEx: Track latency for diagnostics.
  try {
    const channel = resolveEmbeddingChannel({ speaker: normalizedAgent }); // CODEx: Snapshot provider details for logging.
    console.log('[Embedding Test] starting', { agent: normalizedAgent, provider: channel.provider, model: channel.model, bucket: channel.bucket }); // CODEx: Emit telemetry prior to request.
    const vector = await generateTextEmbedding(sampleText, { speaker: normalizedAgent }); // CODEx: Reuse standard embedding flow scoped to agent bucket.
    const duration = Math.max(1, getTimestampMs() - started); // CODEx: Derive latency in milliseconds.
    const length = Array.isArray(vector) ? vector.length : 0; // CODEx: Capture resulting vector dimensionality.
    if (statusElement) {
      statusElement.textContent = `✓ ${length} dims in ${duration} ms`; // CODEx: Surface successful metrics inline.
    }
    console.log('[Embedding Test] success', { agent: normalizedAgent, provider: channel.provider, model: channel.model, duration, dimensions: length }); // CODEx: Persist success telemetry.
  } catch (error) {
    if (statusElement) {
      statusElement.textContent = `✗ ${error?.message ?? 'Failed'}`; // CODEx: Surface error message inline for the user.
    }
    console.warn('[Embedding Test] failure', { agent: normalizedAgent, error }); // CODEx: Emit warning for debugging.
  }
} // CODEx

function estimateEmbeddingBytes(vector) { // CODEx: Estimate bytes consumed by an embedding array.
  if (!Array.isArray(vector)) { // CODEx: Ignore malformed embedding payloads.
    return 0; // CODEx: Count no memory usage when vectors are invalid.
  } // CODEx
  return vector.length * 8; // CODEx: Assume double-precision floats for footprint approximation.
} // CODEx

function storeChunkEmbedding(key, vector, bucketKey = EMBEDDING_BUCKET_KEYS.SHARED) { // CODEx: Cache chunk embeddings per bucket.
  if (!Array.isArray(vector) || !vector.length) { // CODEx: Skip invalid vectors.
    return; // CODEx: Avoid storing malformed embeddings.
  }
  const normalized = normalizeEmbeddingBucket(bucketKey); // CODEx: Normalize bucket identifier.
  const bucket = getRuntimeEmbeddingBucket(normalized); // CODEx: Access runtime state for the bucket.
  const previous = bucket.index.get(key); // CODEx: Retrieve prior embedding for footprint adjustments.
  if (!bucket.index.has(key)) { // CODEx: Track insertion order for new entries.
    bucket.order.push({ key, bucket: normalized }); // CODEx: Record entry metadata for eviction.
  }
  if (Array.isArray(previous)) { // CODEx: Remove prior footprint before writing new vector.
    const previousBytes = estimateEmbeddingBytes(previous); // CODEx: Measure previous contribution.
    bucket.footprint = Math.max(0, bucket.footprint - previousBytes); // CODEx: Update bucket footprint.
    totalEmbeddingFootprintBytes = Math.max(0, totalEmbeddingFootprintBytes - previousBytes); // CODEx: Update aggregate footprint.
  }
  bucket.index.set(key, vector); // CODEx: Persist vector in-memory for the bucket.
  const addedBytes = estimateEmbeddingBytes(vector); // CODEx: Measure new contribution.
  bucket.footprint += addedBytes; // CODEx: Update bucket footprint.
  totalEmbeddingFootprintBytes += addedBytes; // CODEx: Update aggregate footprint.
  enforceChunkEmbeddingBudget(normalized); // CODEx: Ensure footprints remain within configured limits.
  persistEmbeddingVector(normalized, key, vector); // CODEx: Mirror embedding into disk-backed cache per bucket.
} // CODEx

function enforceChunkEmbeddingBudget(bucketKey = EMBEDDING_BUCKET_KEYS.SHARED) { // CODEx: Trim cached embeddings when exceeding budgets.
  const normalized = normalizeEmbeddingBucket(bucketKey); // CODEx: Normalize identifier.
  const bucket = getRuntimeEmbeddingBucket(normalized); // CODEx: Access bucket runtime state.
  if (totalEmbeddingFootprintBytes <= MAX_RAG_EMBEDDING_BYTES) { // CODEx: Skip trimming when global footprint is safe.
    return; // CODEx: No action required under limit.
  }
  let removed = 0; // CODEx: Track total evictions for logging.
  while (totalEmbeddingFootprintBytes > MAX_RAG_EMBEDDING_BYTES && bucket.order.length) { // CODEx: Continue trimming until within limit.
    const oldest = bucket.order.shift(); // CODEx: Pop oldest entry metadata.
    const vector = bucket.index.get(oldest.key); // CODEx: Retrieve stored vector for eviction.
    if (!Array.isArray(vector)) { // CODEx: Skip entries already cleared.
      continue; // CODEx: Proceed to next candidate.
    }
    const bytes = estimateEmbeddingBytes(vector); // CODEx: Measure removed footprint.
    bucket.index.delete(oldest.key); // CODEx: Remove from runtime cache.
    bucket.footprint = Math.max(0, bucket.footprint - bytes); // CODEx: Adjust bucket footprint.
    totalEmbeddingFootprintBytes = Math.max(0, totalEmbeddingFootprintBytes - bytes); // CODEx: Adjust aggregate footprint.
    removed += 1; // CODEx: Increment eviction counter.
  }
  if (removed > 0) { // CODEx: Emit diagnostics when evictions occur.
    console.debug('Embedding cache trimmed', { bucket: normalized, removed, remainingBytes: bucket.footprint }); // CODEx: Log trimming metrics.
  }
} // CODEx

function getEmbeddingCachePath(bucketKey, entryKey) { // CODEx: Map cache key to disk filename per bucket.
  const directory = EMBEDDING_CACHE_DIRECTORIES[normalizeEmbeddingBucket(bucketKey)]; // CODEx: Resolve directory for bucket.
  const hash = hashString(entryKey || ''); // CODEx: Hash key for filesystem safety.
  return `${directory}/${hash}.json`; // CODEx: Derive deterministic cache path.
} // CODEx

function updateManifestEntry(entry, bytes) { // CODEx: Update entry metadata with new byte counts.
  const now = Date.now(); // CODEx: Timestamp for LRU ordering.
  entry.bytes = bytes; // CODEx: Refresh entry footprint.
  entry.updatedAt = now; // CODEx: Track update timestamp.
  entry.lastAccess = now; // CODEx: Refresh access timestamp when writing.
} // CODEx

function trimEmbeddingCacheManifest(bucketKey = EMBEDDING_BUCKET_KEYS.SHARED) { // CODEx: Evict least-recently-used entries per bucket.
  const normalized = normalizeEmbeddingBucket(bucketKey); // CODEx: Normalize identifier.
  const manifest = getEmbeddingManifest(normalized); // CODEx: Retrieve manifest for trimming.
  const maxEntries = 10000; // CODEx: Upper bound on cached entries to avoid unbounded manifests.
  if (manifest.totalBytes <= EMBEDDING_CACHE_TRIM_THRESHOLD && manifest.entries.length <= maxEntries) { // CODEx: Skip trimming when within byte and entry limits.
    return; // CODEx: Nothing to evict.
  }
  const sorted = manifest.entries.slice().sort((a, b) => (a.lastAccess || 0) - (b.lastAccess || 0)); // CODEx: Sort ascending by last access.
  for (const entry of sorted) { // CODEx: Remove until total falls below threshold.
    if (manifest.totalBytes <= EMBEDDING_CACHE_TRIM_THRESHOLD && manifest.entries.length <= maxEntries) { // CODEx: Stop when both limits satisfied.
      break; // CODEx: Exit loop once within budget.
    }
    if (entry?.path) { // CODEx: Delete persisted vector when path known.
      deleteStandaloneFile(entry.path); // CODEx: Remove cached vector file.
    }
    manifest.entries = manifest.entries.filter((candidate) => candidate.key !== entry.key); // CODEx: Remove entry from manifest array.
    const bytes = Number.isFinite(entry?.bytes) ? entry.bytes : 0; // CODEx: Normalize byte count.
    manifest.totalBytes = Math.max(0, manifest.totalBytes - bytes); // CODEx: Adjust manifest footprint.
  }
  persistEmbeddingCacheManifest(normalized); // CODEx: Persist manifest updates after trimming.
} // CODEx

function persistEmbeddingVector(bucketKey, entryKey, vector) { // CODEx: Store embedding vector to disk cache per bucket.
  if (!Array.isArray(vector) || !vector.length) { // CODEx: Skip invalid vectors.
    return; // CODEx: Do not persist malformed embeddings.
  }
  const normalized = normalizeEmbeddingBucket(bucketKey); // CODEx: Normalize identifier for persistence.
  const manifest = getEmbeddingManifest(normalized); // CODEx: Retrieve manifest snapshot.
  try { // CODEx: Serialize vector payload for disk storage.
    const directory = EMBEDDING_CACHE_DIRECTORIES[normalized]; // CODEx: Directory target for bucket.
    ensureStandaloneDirectory(directory); // CODEx: Ensure directory exists before writing.
    const path = getEmbeddingCachePath(normalized, entryKey); // CODEx: Determine file path for key.
    const payload = JSON.stringify({ key: entryKey, vector }); // CODEx: Compose cache entry payload.
    const bytes = new TextEncoder().encode(payload).length; // CODEx: Measure serialized byte length.
    const saved = writeStandaloneFile(path, payload); // CODEx: Persist vector to disk.
    if (saved === false) { // CODEx: Abort caching when filesystem bridge is unavailable.
      return; // CODEx: Respect environments without disk persistence.
    }
    const index = manifest.entries.findIndex((entry) => entry.key === entryKey); // CODEx: Locate prior entry.
    if (index >= 0) { // CODEx: Update existing entry.
      const existing = manifest.entries[index]; // CODEx: Reference existing manifest entry.
      const previousBytes = Number.isFinite(existing.bytes) ? existing.bytes : 0; // CODEx: Previous footprint for adjustments.
      manifest.totalBytes = Math.max(0, manifest.totalBytes - previousBytes); // CODEx: Remove previous byte contribution.
      updateManifestEntry(existing, bytes); // CODEx: Refresh entry metadata.
      existing.path = path; // CODEx: Store path for reloading.
    } else { // CODEx: Create new manifest entry when absent.
      manifest.entries.push({ key: entryKey, path, bytes, updatedAt: Date.now(), lastAccess: Date.now() }); // CODEx: Append manifest entry.
    }
    manifest.totalBytes += bytes; // CODEx: Add new byte contribution.
    trimEmbeddingCacheManifest(normalized); // CODEx: Enforce disk budget with LRU trimming.
    persistEmbeddingCacheManifest(normalized); // CODEx: Flush manifest updates to disk.
  } catch (error) { // CODEx: Handle serialization or IO issues gracefully.
    console.warn('Failed to persist embedding vector', { bucket: normalized, key: entryKey, error }); // CODEx: Warn when caching fails.
  }
} // CODEx

function loadEmbeddingFromDisk(bucketKey, entryKey) { // CODEx: Retrieve persisted vector for cache misses.
  const normalized = normalizeEmbeddingBucket(bucketKey); // CODEx: Normalize identifier.
  const manifest = getEmbeddingManifest(normalized); // CODEx: Retrieve manifest entries.
  const entry = manifest.entries.find((candidate) => candidate.key === entryKey); // CODEx: Locate manifest entry by key.
  if (!entry || !entry.path) { // CODEx: Abort when entry missing.
    return null; // CODEx: Disk cache miss.
  }
  const record = readStandaloneFile(entry.path); // CODEx: Read cached file via bridge.
  if (!record?.content) { // CODEx: Skip when file missing.
    return null; // CODEx: Disk cache miss due to missing file.
  }
  try { // CODEx: Parse cached payload.
    const parsed = JSON.parse(record.content); // CODEx: Interpret JSON content.
    const vector = Array.isArray(parsed?.vector) ? parsed.vector : Array.isArray(parsed?.embedding) ? parsed.embedding : null; // CODEx: Accept legacy shapes.
    if (Array.isArray(vector) && vector.length) { // CODEx: Validate vector.
      entry.lastAccess = Date.now(); // CODEx: Refresh access timestamp on hit.
      persistEmbeddingCacheManifest(normalized); // CODEx: Persist updated access time for LRU ordering.
      return vector; // CODEx: Return hydrated vector.
    }
  } catch (error) { // CODEx: Handle JSON parse errors gracefully.
    console.warn('Failed to parse cached embedding vector', { bucket: normalized, key: entryKey, error }); // CODEx: Emit parse warning.
  }
  return null; // CODEx: Default to null when payload invalid.
}

function resolveCachedEmbedding(bucketKey, entryKey) { // CODEx: Fetch embedding from runtime cache or disk fallback.
  const normalized = normalizeEmbeddingBucket(bucketKey); // CODEx: Normalize identifier for lookups.
  const bucket = getRuntimeEmbeddingBucket(normalized); // CODEx: Access in-memory cache for the bucket.
  const cached = bucket.index.get(entryKey); // CODEx: Attempt to read from runtime cache first.
  if (Array.isArray(cached) && cached.length) { // CODEx: Return immediately when runtime cache hit.
    return cached; // CODEx: Provide cached embedding without disk access.
  }
  return loadEmbeddingFromDisk(normalized, entryKey); // CODEx: Fallback to persisted embedding when runtime miss occurs.
} // CODEx

function resolveEmbeddingContextLength(model, override) { // CODEx: Derive context window based on model hints and user override.
  const baseOverride = Number.isFinite(override) ? override : config.embeddingContextLength ?? defaultConfig.embeddingContextLength; // CODEx: Prefer configured window when numeric.
  let hinted = 1024; // CODEx: Default context size when hints absent.
  if (typeof model === 'string') { // CODEx: Inspect model identifier for context hints.
    if (/4096/i.test(model)) { // CODEx: Detect 4k context models.
      hinted = 4096;
    } else if (/2048/i.test(model)) { // CODEx: Detect 2k context models.
      hinted = 2048;
    }
  }
  const candidate = Math.max(hinted, baseOverride || hinted); // CODEx: Choose the larger of hint or override.
  return clampNumber(candidate, 512, 4096, 1024); // CODEx: Enforce 512–4096 bounds with 1024 fallback.
} // CODEx

function resolveEmbeddingChannel(options = {}) { // CODEx: Map speakers to embedding bucket, provider, and model.
  const speaker = typeof options.speaker === 'string' ? options.speaker.toUpperCase() : ''; // CODEx: Normalize speaker identifier.
  const inferredBucket = speaker === 'B' ? EMBEDDING_BUCKET_KEYS.SAM_B : speaker === 'A' ? EMBEDDING_BUCKET_KEYS.SAM_A : null; // CODEx: Map speaker to bucket.
  const desiredBucket = normalizeEmbeddingBucket(options.bucketKey ?? inferredBucket ?? EMBEDDING_BUCKET_KEYS.SHARED); // CODEx: Resolve final bucket selection.
  if (desiredBucket === EMBEDDING_BUCKET_KEYS.SAM_A) { // CODEx: SAM-A specific channel configuration.
    const provider = config.agentAEmbeddingProvider || defaultConfig.agentAEmbeddingProvider; // CODEx: Prefer configured SAM-A provider.
    const model = (config.agentAEmbeddingModel || defaultConfig.agentAEmbeddingModel).trim(); // CODEx: Normalize SAM-A model.
    const contextLength = resolveEmbeddingContextLength(model, config.agentAEmbeddingContextLength); // CODEx: Derive SAM-A context window.
    const endpoint = (config.agentAEmbeddingEndpoint || '').trim(); // CODEx: SAM-A endpoint override when provided.
    const apiKey = (config.agentAEmbeddingApiKey || '').trim(); // CODEx: SAM-A API key override when provided.
    const baseUrl = endpoint ? deriveBaseOrigin(endpoint) : ''; // CODEx: Derive SAM-A base URL for provider bridge.
    return { bucket: desiredBucket, provider, model, contextLength, endpoint, apiKey, baseUrl, useDetection: false, speaker: 'A' }; // CODEx: Return SAM-A channel descriptor.
  }
  if (desiredBucket === EMBEDDING_BUCKET_KEYS.SAM_B) { // CODEx: SAM-B specific channel configuration.
    const provider = config.agentBEmbeddingProvider || defaultConfig.agentBEmbeddingProvider; // CODEx: Prefer configured SAM-B provider.
    const model = (config.agentBEmbeddingModel || defaultConfig.agentBEmbeddingModel).trim(); // CODEx: Normalize SAM-B model.
    const contextLength = resolveEmbeddingContextLength(model, config.agentBEmbeddingContextLength); // CODEx: Derive SAM-B context window.
    const endpoint = (config.agentBEmbeddingEndpoint || '').trim(); // CODEx: SAM-B endpoint override when provided.
    const apiKey = (config.agentBEmbeddingApiKey || '').trim(); // CODEx: SAM-B API key override when provided.
    const baseUrl = endpoint ? deriveBaseOrigin(endpoint) : ''; // CODEx: Derive SAM-B base URL for provider bridge.
    return { bucket: desiredBucket, provider, model, contextLength, endpoint, apiKey, baseUrl, useDetection: false, speaker: 'B' }; // CODEx: Return SAM-B channel descriptor.
  }
  const model = (config.embeddingModel || defaultConfig.embeddingModel || 'text-embedding-3-large').trim(); // CODEx: Resolve shared embedding model.
  const contextLength = resolveEmbeddingContextLength(model, config.embeddingContextLength); // CODEx: Derive shared context window.
  return {
    bucket: desiredBucket,
    provider: config.embeddingProviderPreference || defaultConfig.embeddingProviderPreference, // CODEx: Shared provider preference.
    model,
    contextLength,
    endpoint: (config.embeddingEndpoint || '').trim(), // CODEx: Shared endpoint override when provided.
    apiKey: (config.embeddingApiKey || config.apiKey || '').trim(), // CODEx: Shared embedding API key fallback.
    baseUrl: config.embeddingEndpoint ? deriveBaseOrigin(config.embeddingEndpoint) : '', // CODEx: Shared base URL hint for provider bridge.
    useDetection: true,
    speaker: speaker || null
  }; // CODEx: Shared embedding channel descriptor.
} // CODEx

async function warmEmbeddings() { // CODEx: Pre-initialize embedding channels to reduce first-turn latency.
  if (!isEmbeddingRetrievalEnabled()) { // CODEx: Skip warmup when embeddings disabled.
    return;
  }
  const warmTargets = [ // CODEx: Define arena-specific warmup targets.
    {
      name: 'Shared',
      bucket: EMBEDDING_BUCKET_KEYS.SHARED,
      speaker: null,
      provider: config.embeddingProviderPreference || defaultConfig.embeddingProviderPreference,
      model: (config.embeddingModel || defaultConfig.embeddingModel)
    },
    {
      name: 'SAM-A',
      bucket: EMBEDDING_BUCKET_KEYS.SAM_A,
      speaker: 'A',
      provider: config.agentAEmbeddingProvider || defaultConfig.agentAEmbeddingProvider,
      model: config.agentAEmbeddingModel || defaultConfig.agentAEmbeddingModel
    },
    {
      name: 'SAM-B',
      bucket: EMBEDDING_BUCKET_KEYS.SAM_B,
      speaker: 'B',
      provider: config.agentBEmbeddingProvider || defaultConfig.agentBEmbeddingProvider,
      model: config.agentBEmbeddingModel || defaultConfig.agentBEmbeddingModel
    }
  ];
  const tasks = warmTargets.map(async (target) => { // CODEx: Batch warmup requests concurrently.
    const start = getTimestampMs();
    try {
      console.log(`[Warmup] ${target.name} → ${target.provider}`); // CODEx: Emit warmup telemetry.
      await generateTextEmbedding(`Warmup vector init (${target.model})`, { bucketKey: target.bucket, speaker: target.speaker }); // CODEx: Prime channel embeddings.
      const duration = Math.round(Math.max(1, getTimestampMs() - start));
      return { status: 'ok', name: target.name, duration };
    } catch (error) {
      console.warn(`[Warmup] ${target.name} failed`, error); // CODEx: Surface warmup failures but continue.
      return { status: 'error', name: target.name, error: error?.message || String(error) };
    }
  });
  const results = await Promise.allSettled(tasks); // CODEx: Await completion without aborting on errors.
  const successes = results
    .map((entry) => (entry.status === 'fulfilled' ? entry.value : entry.reason))
    .filter((entry) => entry && entry.status === 'ok'); // CODEx: Collect successful warmups.
  if (successes.length) {
    const average = successes.reduce((sum, entry) => sum + entry.duration, 0) / successes.length; // CODEx: Compute average latency.
    const rounded = Math.round(average);
    console.log('[Perf] avg embed ms', rounded); // CODEx: Report warmup average to console telemetry.
    console.log('[Perf] avg fetch ms', rounded); // CODEx: Mirror fetch timing metric for diagnostics.
  }
} // CODEx
async function resolveActiveEmbeddingProvider(force = false) { // CODEx: Probe embedding provider based on preference and connectivity.
  const preference = config.embeddingProviderPreference ?? defaultConfig.embeddingProviderPreference; // CODEx: Read preference from config.
  embeddingProviderState.preference = preference; // CODEx: Track preference for status UI.
  const now = Date.now(); // CODEx: Timestamp to throttle probes.
  const stale = force || !embeddingProviderState.lastProbe || now - embeddingProviderState.lastProbe > 60_000; // CODEx: Re-probe every minute or on demand.
  if (!stale && embeddingProviderState.active) { // CODEx: Skip detection when provider state fresh.
    return { ...embeddingProviderState }; // CODEx: Return cached provider snapshot.
  }
  try { // CODEx: Attempt detection across local and remote providers.
    const detection = await detectEmbeddingService({
      preference,
      embeddingEndpoint: config.embeddingEndpoint,
      modelEndpoint: config.endpoint,
      embeddingApiKey: (config.embeddingApiKey || config.apiKey || '').trim(), // CODEx: Pass configured API keys for provider detection.
      fetchImpl: (url, options) => fetch(url, options),
      logger: console
    });
    embeddingProviderState.active = detection.provider ?? EMBEDDING_PROVIDERS.OPENAI; // CODEx: Persist detected provider.
    embeddingProviderState.baseUrl = detection.baseUrl ?? ''; // CODEx: Persist associated base URL.
    embeddingProviderState.lastProbe = now; // CODEx: Record probe time.
  } catch (error) { // CODEx: Fallback to remote provider when detection fails.
    console.warn('Embedding provider detection failed', error); // CODEx: Emit diagnostic warning.
    embeddingProviderState.active = EMBEDDING_PROVIDERS.OPENAI; // CODEx: Default to remote embeddings.
    embeddingProviderState.baseUrl = ''; // CODEx: Clear base URL on failure.
    embeddingProviderState.lastProbe = now; // CODEx: Avoid repeated rapid probes.
  }
  refreshEmbeddingStatusIndicator(); // CODEx: Update status indicator with latest provider.
  return { ...embeddingProviderState }; // CODEx: Return provider snapshot for callers.
}
const phasePromptCache = new Map();
let activePhaseIndex = 0;
let activePhaseId = null;
let pendingPhaseAdvance = false;
let lastReflexSynthesisSignature = null;
let lastWatchdogArchiveAt = 0;

function getManifestStorageKey(descriptor) {
  if (!descriptor) return `${RAM_DISK_MANIFEST_PREFIX}default`;
  const base = descriptor.ramKey || descriptor.url || 'manifest';
  return `${RAM_DISK_MANIFEST_PREFIX}${base}`;
}

function cacheManifestInRam(descriptor, manifest) {
  if (!descriptor?.url || !manifest) return;
  ramDiskCache.manifests.set(descriptor.url, manifest);
}

function readManifestFromRam(descriptor) {
  if (!descriptor?.url) return null;
  return ramDiskCache.manifests.get(descriptor.url) || null;
}

function persistManifestToStorage(descriptor, manifest) {
  if (!manifest) return;
  try {
    const key = getManifestStorageKey(descriptor);
    storageSetItem(key, JSON.stringify(manifest));
  } catch (error) {
    console.warn('Failed to persist manifest to RAM disk storage:', error);
  }
}

function readManifestFromStorage(descriptor) {
  try {
    const key = getManifestStorageKey(descriptor);
    const stored = storageGetItem(key);
    if (!stored) return null;
    const parsed = JSON.parse(stored);
    cacheManifestInRam(descriptor, parsed);
    return parsed;
  } catch (error) {
    console.warn('Failed to read manifest from RAM disk storage:', error);
    return null;
  }
}
let chunkerState = {
  chunkSize: 500,
  chunkOverlap: 50,
  unchunkedFiles: [],
  chunkedFiles: [],
  totalChunks: 0,
  isProcessing: false
};
let processRegistry = new Map();
let debugPanelVisible = false;
let modelAInFlight = false;
let modelBInFlight = false;
let pdfLoaderPromise;
const PROCESS_BASELINE = [
  { id: 'modelA', label: 'Model A pipeline', status: 'Idle', detail: 'Waiting for user prompt.' },
  { id: 'modelB', label: 'Model B pipeline', status: 'Idle', detail: 'Awaiting arena start.' },
  { id: 'arena', label: 'SAM arena loop', status: 'Idle', detail: 'Dual chat stopped.' },
  { id: 'autosave', label: 'Arena autosave', status: 'Idle', detail: 'Next snapshot pending.' }
];

const elements = {
  chatWindow: document.getElementById('chatWindow'),
  dualChatWindow: document.getElementById('dualChatWindow'),
  entropyBarFill: document.getElementById('entropyBarFill'),
  entropyStatus: document.getElementById('entropyStatus'),
  reflexStatus: document.getElementById('reflexStatus'),
  modelStatus: document.getElementById('modelStatus'),
  modelBStatus: document.getElementById('modelBStatus'),
  ragStatusFlag: document.getElementById('ragStatusFlag'),
  memoryStatus: document.getElementById('memoryStatus'),
  systemResourceStatus: document.getElementById('systemResourceStatus'),
  optionsHandle: document.getElementById('optionsHandle'),
  optionsBackdrop: document.getElementById('optionsBackdrop'),
  optionsDrawer: document.getElementById('optionsDrawer'),
  modeOverlay: document.getElementById('modeOverlay'),
  enterChatButton: document.getElementById('enterChatButton'),
  enterArenaButton: document.getElementById('enterArenaButton'),
  statusDock: document.getElementById('statusDock'),
  chatPanel: document.getElementById('chatPanel'),
  arenaPanel: document.getElementById('arenaPanel'),
  switchToChatButton: document.getElementById('switchToChatButton'),
  switchToArenaButton: document.getElementById('switchToArenaButton'),
  agentACard: document.getElementById('agentACard'), // CODEx: Agent A card container.
  agentABody: document.getElementById('agentABody'), // CODEx: Agent A body panel.
  toggleAgentA: document.getElementById('toggleAgentA'), // CODEx: Agent A collapse toggle.
  agentBCard: document.getElementById('agentBCard'), // CODEx: Agent B card container.
  agentBBody: document.getElementById('agentBBody'), // CODEx: Agent B body panel.
  toggleAgentB: document.getElementById('toggleAgentB'), // CODEx: Agent B collapse toggle.
  agentBEnabled: document.getElementById('agentBEnabled'),
  messageInput: document.getElementById('messageInput'),
  sendButton: document.getElementById('sendButton'),
  voiceButton: document.getElementById('voiceButton'),
  speakToggle: document.getElementById('speakToggle'),
  retrieveButton: document.getElementById('retrieveButton'),
  clearChatButton: document.getElementById('clearChatButton'),
  memorySlider: document.getElementById('memorySlider'),
  memorySliderValue: document.getElementById('memorySliderValue'),
  retrievalCount: document.getElementById('retrievalCount'),
  reasoningModeToggle: document.getElementById('reasoningModeToggle'),
  embeddingRetrievalToggle: document.getElementById('embeddingRetrievalToggle'), // CODEx: Toggle element for embedding-based retrieval.
  embeddingStatus: document.getElementById('embeddingStatus'), // CODEx: Status label showing embedding availability.
  autoInjectMemories: document.getElementById('autoInjectMemories'),
  contextTurns: document.getElementById('contextTurns'),
  exportButton: document.getElementById('exportMemoryButton'),
  floatingMemoryList: document.getElementById('floatingMemoryList'),
  floatingMemoryCount: document.getElementById('floatingMemoryCount'),
  pinnedMemoryCount: document.getElementById('pinnedMemoryCount'),
  ragMemoryCount: document.getElementById('ragMemoryCount'),
  ragSnapshotCount: document.getElementById('ragSnapshotCount'),
  ragFootprint: document.getElementById('ragFootprint'),
  ragCompressionRatio: document.getElementById('ragCompressionRatio'),
  ragImportStatus: document.getElementById('ragImportStatus'),
  refreshFloatingButton: document.getElementById('refreshFloatingButton'),
  loadRagButton: document.getElementById('loadRagButton'),
  loadChunkDataButton: document.getElementById('loadChunkDataButton'),
  saveChatArchiveButton: document.getElementById('saveChatArchiveButton'),
  archiveUnpinnedButton: document.getElementById('archiveUnpinnedButton'),
  modelARetrievals: document.getElementById('modelARetrievals'),
  modelBRetrievals: document.getElementById('modelBRetrievals'),
  providerSelect: document.getElementById('a_model_preset'), // CODEx: Agent A provider preset selector.
  providerNotes: document.getElementById('providerNotes'), // CODEx: Agent A provider notes output.
  endpointInput: document.getElementById('a_model_endpoint'), // CODEx: Agent A endpoint input.
  modelInput: document.getElementById('a_model_name'), // CODEx: Agent A model name input.
  apiKeyInput: document.getElementById('a_model_key'), // CODEx: Agent A API key input.
  temperatureInput: document.getElementById('a_temp'), // CODEx: Agent A temperature slider.
  temperatureValueLabel: document.getElementById('a_temp_value'), // CODEx: Agent A temperature label.
  maxTokensInput: document.getElementById('a_max_tokens'), // CODEx: Agent A max tokens input.
  agentAEmbeddingProviderSelect: document.getElementById('a_embed_provider'), // CODEx: Agent A embedding provider select.
  agentAEmbeddingModelInput: document.getElementById('a_embed_model'), // CODEx: Agent A embedding model input.
  agentAEmbeddingEndpointInput: document.getElementById('a_embed_endpoint'), // CODEx: Agent A embedding endpoint input.
  agentAEmbeddingApiKeyInput: document.getElementById('a_embed_key'), // CODEx: Agent A embedding key input.
  agentAEmbeddingContextInput: document.getElementById('a_embed_ctx'), // CODEx: Agent A embedding context input.
  agentAEmbeddingTestButton: document.getElementById('a_test_embed'), // CODEx: Agent A embedding test button.
  agentAEmbeddingStatus: document.getElementById('a_embed_status'), // CODEx: Agent A embedding status label.
  agentBEmbeddingProviderSelect: document.getElementById('b_embed_provider'), // CODEx: Agent B embedding provider select.
  agentBEmbeddingModelInput: document.getElementById('b_embed_model'), // CODEx: Agent B embedding model input.
  agentBEmbeddingEndpointInput: document.getElementById('b_embed_endpoint'), // CODEx: Agent B embedding endpoint input.
  agentBEmbeddingApiKeyInput: document.getElementById('b_embed_key'), // CODEx: Agent B embedding key input.
  agentBEmbeddingContextInput: document.getElementById('b_embed_ctx'), // CODEx: Agent B embedding context input.
  agentBEmbeddingTestButton: document.getElementById('b_test_embed'), // CODEx: Agent B embedding test button.
  agentBEmbeddingStatus: document.getElementById('b_embed_status'), // CODEx: Agent B embedding status label.
  saveConfigButton: document.getElementById('saveOptions'), // CODEx: Manual save control.
  saveStatus: document.getElementById('saveStatus'), // CODEx: Save feedback label.
  openRouterPolicyField: document.getElementById('openRouterPolicyField'),
  openRouterPolicyInput: document.getElementById('openRouterPolicyInput'),
  systemPromptInput: document.getElementById('systemPromptInput'),
  maxTokensHint: document.getElementById('maxTokensHint'),
  diagnosticsButton: document.getElementById('diagnosticsButton'),
  dualStatus: document.getElementById('dualStatus'),
  startDualButton: document.getElementById('startDualButton'),
  nextDualButton: document.getElementById('nextDualButton'),
  stopDualButton: document.getElementById('stopDualButton'),
  exportDualButton: document.getElementById('exportDualButton'),
  dualSeedInput: document.getElementById('dualSeedInput'),
  dualTurnLimit: document.getElementById('dualTurnLimit'),
  dualTurnDelay: document.getElementById('dualTurnDelay'),
  dualAutoContinue: document.getElementById('dualAutoContinue'),
  reflexToggle: document.getElementById('reflexToggle'),
  reflexInterval: document.getElementById('reflexInterval'),
  entropyWindow: document.getElementById('entropyWindow'),
  triggerReflexButton: document.getElementById('triggerReflexButton'),
  reflexStatusText: document.getElementById('reflexStatusText'),
  agentAName: document.getElementById('agentAName'),
  agentBName: document.getElementById('agentBName'),
  agentBPrompt: document.getElementById('agentBPrompt'),
  requestTimeout: document.getElementById('requestTimeout'),
  reasoningTimeout: document.getElementById('reasoningTimeout'),
  agentBProviderSelect: document.getElementById('b_model_preset'), // CODEx: Agent B provider preset selector.
  agentBEndpoint: document.getElementById('b_model_endpoint'), // CODEx: Agent B endpoint input.
  agentBModel: document.getElementById('b_model_name'), // CODEx: Agent B model input.
  agentBApiKey: document.getElementById('b_model_key'), // CODEx: Agent B API key input.
  agentBTemperatureInput: document.getElementById('b_temp'), // CODEx: Agent B temperature slider.
  agentBTemperatureLabel: document.getElementById('b_temp_value'), // CODEx: Agent B temperature value label.
  agentBMaxTokensInput: document.getElementById('b_max_tokens'), // CODEx: Agent B max tokens input.
  ttsPresetSelect: document.getElementById('ttsPresetSelect'),
  ttsPresetDetails: document.getElementById('ttsPresetDetails'),
  ttsPresetList: document.getElementById('ttsPresetList'),
  ttsServerField: document.querySelector('[data-tts-field="server"]'),
  ttsVoiceField: document.querySelector('[data-tts-field="voice"]'),
  ttsApiKeyField: document.querySelector('[data-tts-field="apiKey"]'),
  ttsServerUrl: document.getElementById('ttsServerUrl'),
  ttsVoiceId: document.getElementById('ttsVoiceId'),
  ttsVoiceCatalog: document.getElementById('ttsVoiceCatalog'),
  ttsApiKeyInput: document.getElementById('ttsApiKeyInput'),
  ttsVolume: document.getElementById('ttsVolume'),
  ttsVolumeValue: document.getElementById('ttsVolumeValue'),
  testTtsButton: document.getElementById('testTtsButton'),
  backgroundUrlInput: document.getElementById('backgroundUrlInput'),
  backgroundFileInput: document.getElementById('backgroundFileInput'),
  clearBackgroundButton: document.getElementById('clearBackgroundButton'),
  backgroundStatus: document.getElementById('backgroundStatus'),
  messageTemplate: document.getElementById('messageTemplate'),
  dualMessageTemplate: document.getElementById('dualMessageTemplate'),
  debugToggle: document.getElementById('debugToggle'),
  openDebugButton: document.getElementById('openDebugButton'),
  debugPanel: document.getElementById('debugPanel'),
  debugCloseButton: document.getElementById('debugCloseButton'),
  processList: document.getElementById('processList'),
  logList: document.getElementById('logList'),
  logStatus: document.getElementById('logStatus'),
  logShutdownButton: document.getElementById('logShutdownButton'),
  logErrorButton: document.getElementById('logErrorButton'),
  exportLogsButton: document.getElementById('exportLogsButton'),
  chunkSizeInput: document.getElementById('chunkSizeInput'),
  chunkOverlapInput: document.getElementById('chunkOverlapInput'),
  chunkerStatus: document.getElementById('chunkerStatus'),
  unchunkedFileCount: document.getElementById('unchunkedFileCount'),
  chunkedFileCount: document.getElementById('chunkedFileCount'),
  totalChunkCount: document.getElementById('totalChunkCount'),
  scanChunkerButton: document.getElementById('scanChunkerButton'),
  chunkFilesButton: document.getElementById('chunkFilesButton'),
  clearChunksButton: document.getElementById('clearChunksButton')
};

init();

async function init() {
  closeOptionsPanel();
  document.body.dataset.mode = MODE_CHOOSER;
  loadConfig();
  applyBackgroundFromConfig();
  loadPinnedMessages();
  loadRetrievalMetrics();
  initProcessRegistry();
  populateProviderSelect();
  populateArenaProviderSelects();
  populateTtsControls();
  bindEvents();
  ensureEmbeddingCacheManifest(); // CODEx: Hydrate embedding cache manifest before retrieval operations.
  await resolveActiveEmbeddingProvider(true); // CODEx: Probe embedding provider at startup for accurate status.
  void warmEmbeddings(); // CODEx: Pre-warm embedding channels asynchronously.
  primeRamDiskCache();
  await ensurePhasePromptBaseline(); // CODEx: Confirm arena phase prompts are provisioned before debates begin.
  applyDebugSetting();
  updateConfigInputs();
  updateSpeakToggle();
  updateModelConnectionStatus();
  ensureRagSessionId(MODE_CHAT);
  ensureRagSessionId(MODE_ARENA);
  const responsiveQuery = window.matchMedia('(max-width: 1100px)');
  responsiveQuery.addEventListener('change', (event) => {
    if (!event.matches) {
      closeOptionsPanel();
    }
  });
  await initDatabase();
  await loadLogsFromStorage();
  await recordLog('startup', 'SAM workspace initialized.');
  await loadConversationFromStorage();
  await hydrateFloatingMemoryFromRag();
  updateMemoryStatus();
  renderFloatingMemoryWorkbench();
  updateReflexStatus();
  updateEntropyMeter();
  updateChunkerStatus();
  addSystemMessage('SAM is ready. Configure Model A to enable live responses and arena debates.');
  setActiveMode(MODE_CHOOSER);
  window.addEventListener('message', handleStandaloneWatchdogMessage);
  window.addEventListener('beforeunload', () => {
    void recordLog('shutdown', 'Browser session closed.', { level: 'info', silent: true });
  });
}

function setActiveMode(mode) {
  const allowedModes = [MODE_CHOOSER, MODE_CHAT, MODE_ARENA];
  const targetMode = allowedModes.includes(mode) ? mode : MODE_CHAT;
  activeMode = targetMode;
  document.body.dataset.mode = targetMode;

  const isChooser = targetMode === MODE_CHOOSER;
  if (elements.modeOverlay) {
    elements.modeOverlay.hidden = !isChooser;
    elements.modeOverlay.setAttribute('aria-hidden', isChooser ? 'false' : 'true');
    if (isChooser && elements.enterChatButton) {
      elements.enterChatButton.focus();
    }
  }
  if (elements.chatPanel) {
    elements.chatPanel.hidden = targetMode !== MODE_CHAT;
  }
  if (elements.arenaPanel) {
    elements.arenaPanel.hidden = targetMode !== MODE_ARENA;
  }

  if (targetMode === MODE_CHAT) {
    ensureRagSessionId(MODE_CHAT);
    if (isDualChatRunning) {
      stopDualChat();
    }
  } else if (targetMode === MODE_ARENA) {
    ensureRagSessionId(MODE_ARENA);
    if (!isDualChatRunning) {
      resetDualChat();
    }
  } else {
    closeOptionsPanel();
    if (isDualChatRunning) {
      stopDualChat();
    }
  }
}

function loadConfig() {
  try {
    const stored = storageGetItem('sam-config');
    if (stored) {
      config = { ...config, ...JSON.parse(stored) };
    }
    if (!providerPresetMap.has(config.providerPreset)) {
      config.providerPreset = defaultConfig.providerPreset;
    }
    config.memoryLimitMB = clampNumber(
      config.memoryLimitMB ?? defaultConfig.memoryLimitMB,
      MIN_MEMORY_LIMIT_MB,
      MAX_MEMORY_LIMIT_MB,
      defaultConfig.memoryLimitMB
    );
    if (!config.agentBProviderPreset || (config.agentBProviderPreset !== 'inherit' && !providerPresetMap.has(config.agentBProviderPreset))) {
      config.agentBProviderPreset = defaultConfig.agentBProviderPreset;
    }
    if (typeof config.agentBEnabled !== 'boolean') {
      config.agentBEnabled = defaultConfig.agentBEnabled;
    }
    delete config.agentAProviderPreset;
    delete config.agentAEndpoint;
    delete config.agentAModel;
    delete config.agentAApiKey;
    config.agentBEndpoint = config.agentBEndpoint ?? '';
    config.agentBModel = config.agentBModel ?? '';
    config.agentBApiKey = config.agentBApiKey ?? '';
    if (!ttsPresetMap.has(config.ttsPreset)) {
      config.ttsPreset = defaultConfig.ttsPreset;
    }
    config.ttsVolume = clampNumber(config.ttsVolume ?? defaultConfig.ttsVolume, 0, 200, defaultConfig.ttsVolume);
    userEditedTtsServer = Boolean(config.ttsServerUrl);
    userEditedTtsVoice = Boolean(config.ttsVoiceId);
    const contextLimit = getProviderContextLimit(config.providerPreset);
    const clampedMaxTokens = clampNumber(
      config.maxResponseTokens ?? defaultConfig.maxResponseTokens,
      16,
      contextLimit,
      defaultConfig.maxResponseTokens
    );
    const normalizedAgentAMax = Math.round(clampedMaxTokens);
    config.agentAMaxResponseTokens = normalizedAgentAMax;
    config.maxResponseTokens = normalizedAgentAMax;
    const agentAPresetTemperature = clampNumber(
      config.agentATemperature ?? config.temperature ?? defaultConfig.agentATemperature,
      0,
      2,
      defaultConfig.agentATemperature
    );
    config.agentATemperature = agentAPresetTemperature;
    config.temperature = agentAPresetTemperature;
    const agentBTempFallback = clampNumber(
      config.agentBTemperature ?? agentAPresetTemperature,
      0,
      2,
      agentAPresetTemperature
    );
    config.agentBTemperature = agentBTempFallback;
    const agentBContextLimit = getProviderContextLimit(
      config.agentBProviderPreset === 'inherit' ? config.providerPreset : config.agentBProviderPreset
    );
    const normalizedAgentBMax = Math.round(
      clampNumber(
        config.agentBMaxResponseTokens ?? normalizedAgentAMax,
        16,
        agentBContextLimit,
        normalizedAgentAMax
      )
    );
    config.agentBMaxResponseTokens = normalizedAgentBMax;
    config.reasoningModeEnabled = Boolean(config.reasoningModeEnabled);
    config.useEmbeddingRetrieval = typeof config.useEmbeddingRetrieval === 'boolean' ? config.useEmbeddingRetrieval : defaultConfig.useEmbeddingRetrieval; // CODEx: Normalize embedding toggle with sane default.
    if (!config.embeddingModel || typeof config.embeddingModel !== 'string') {
      config.embeddingModel = defaultConfig.embeddingModel;
    } else {
      config.embeddingModel = config.embeddingModel.trim();
    }
    config.embeddingEndpoint = typeof config.embeddingEndpoint === 'string'
      ? config.embeddingEndpoint.trim()
      : '';
    config.embeddingApiKey = typeof config.embeddingApiKey === 'string'
      ? config.embeddingApiKey.trim()
      : '';
    const allowedEmbeddingProviders = new Set(Object.values(EMBEDDING_PROVIDERS)); // CODEx: Validate provider preference.
    const storedPreference = typeof config.embeddingProviderPreference === 'string'
      ? config.embeddingProviderPreference.toLowerCase().trim()
      : '';
    const normalizedPreference = storedPreference === 'ollama'
      ? EMBEDDING_PROVIDERS.OLLAMA_LOCAL
      : storedPreference; // CODEx: Migrate legacy Ollama preference to explicit local variant.
    config.embeddingProviderPreference = allowedEmbeddingProviders.has(normalizedPreference)
      ? normalizedPreference
      : defaultConfig.embeddingProviderPreference; // CODEx: Default to auto when preference invalid.
    config.embeddingContextLength = clampNumber(
      config.embeddingContextLength ?? defaultConfig.embeddingContextLength,
      512,
      4096,
      defaultConfig.embeddingContextLength
    ); // CODEx: Clamp embedding context window between 512 and 4096 tokens.
    const rawAgentAProvider = typeof config.agentAEmbeddingProvider === 'string'
      ? config.agentAEmbeddingProvider.toLowerCase().trim()
      : '';
    const normalizedAgentAProvider = rawAgentAProvider === 'ollama'
      ? EMBEDDING_PROVIDERS.OLLAMA_LOCAL
      : rawAgentAProvider;
    config.agentAEmbeddingProvider = allowedEmbeddingProviders.has(normalizedAgentAProvider)
      ? normalizedAgentAProvider
      : defaultConfig.agentAEmbeddingProvider; // CODEx: Default SAM-A embedding provider when invalid.
    const rawAgentBProvider = typeof config.agentBEmbeddingProvider === 'string'
      ? config.agentBEmbeddingProvider.toLowerCase().trim()
      : '';
    const normalizedAgentBProvider = rawAgentBProvider === 'ollama'
      ? EMBEDDING_PROVIDERS.OLLAMA_LOCAL
      : rawAgentBProvider;
    config.agentBEmbeddingProvider = allowedEmbeddingProviders.has(normalizedAgentBProvider)
      ? normalizedAgentBProvider
      : defaultConfig.agentBEmbeddingProvider; // CODEx: Default SAM-B embedding provider when invalid.
    config.agentAEmbeddingModel = typeof config.agentAEmbeddingModel === 'string' && config.agentAEmbeddingModel.trim()
      ? config.agentAEmbeddingModel.trim()
      : defaultConfig.agentAEmbeddingModel; // CODEx: Normalize SAM-A embedding model identifier.
    config.agentBEmbeddingModel = typeof config.agentBEmbeddingModel === 'string' && config.agentBEmbeddingModel.trim()
      ? config.agentBEmbeddingModel.trim()
      : defaultConfig.agentBEmbeddingModel; // CODEx: Normalize SAM-B embedding model identifier.
    config.agentAEmbeddingEndpoint = typeof config.agentAEmbeddingEndpoint === 'string'
      ? config.agentAEmbeddingEndpoint.trim()
      : ''; // CODEx: Normalize SAM-A embedding endpoint override.
    config.agentBEmbeddingEndpoint = typeof config.agentBEmbeddingEndpoint === 'string'
      ? config.agentBEmbeddingEndpoint.trim()
      : ''; // CODEx: Normalize SAM-B embedding endpoint override.
    config.agentAEmbeddingApiKey = typeof config.agentAEmbeddingApiKey === 'string'
      ? config.agentAEmbeddingApiKey.trim()
      : ''; // CODEx: Normalize SAM-A embedding API key.
    config.agentBEmbeddingApiKey = typeof config.agentBEmbeddingApiKey === 'string'
      ? config.agentBEmbeddingApiKey.trim()
      : ''; // CODEx: Normalize SAM-B embedding API key.
    const sharedContextFallback = config.embeddingContextLength ?? defaultConfig.embeddingContextLength; // CODEx: Reuse shared context hint as fallback.
    config.agentAEmbeddingContextLength = clampNumber(
      config.agentAEmbeddingContextLength ?? sharedContextFallback,
      512,
      4096,
      sharedContextFallback
    ); // CODEx: Clamp SAM-A embedding context window.
    config.agentBEmbeddingContextLength = clampNumber(
      config.agentBEmbeddingContextLength ?? sharedContextFallback,
      512,
      4096,
      sharedContextFallback
    ); // CODEx: Clamp SAM-B embedding context window.
    if (typeof config.dualTurnLimit !== 'number' || config.dualTurnLimit < 0) {
      config.dualTurnLimit = defaultConfig.dualTurnLimit;
    }
    config.dualTurnDelaySeconds = clampNumber(
      config.dualTurnDelaySeconds ?? defaultConfig.dualTurnDelaySeconds,
      0,
      600,
      defaultConfig.dualTurnDelaySeconds
    );
    const storedSeed = typeof config.dualSeed === 'string' ? config.dualSeed.trim() : '';
    config.dualSeed = storedSeed || DEFAULT_DUAL_SEED;
    const allowedBackgroundSources = new Set(['default', 'url', 'upload']);
    if (!allowedBackgroundSources.has(config.backgroundSource)) {
      config.backgroundSource = defaultConfig.backgroundSource;
    }
    config.backgroundImage = config.backgroundImage ?? '';
    config.backgroundUrl = config.backgroundUrl ?? '';
    if (config.backgroundSource === 'url' && !config.backgroundUrl) {
      config.backgroundSource = config.backgroundImage ? 'upload' : 'default';
    }
    if (config.backgroundSource === 'upload' && !config.backgroundImage) {
      config.backgroundSource = config.backgroundUrl ? 'url' : 'default';
    }
    config.debugEnabled = Boolean(config.debugEnabled);
    config.reflexEnabled = typeof config.reflexEnabled === 'boolean' ? config.reflexEnabled : defaultConfig.reflexEnabled;
    config.reflexInterval = clampNumber(
      config.reflexInterval ?? defaultConfig.reflexInterval,
      2,
      50,
      defaultConfig.reflexInterval
    );
    config.entropyWindow = clampNumber(
      config.entropyWindow ?? defaultConfig.entropyWindow,
      1,
      25,
      defaultConfig.entropyWindow
    );
    config.requestTimeoutSeconds = clampNumber(
      config.requestTimeoutSeconds ?? defaultConfig.requestTimeoutSeconds,
      5,
      600,
      defaultConfig.requestTimeoutSeconds
    );
    const reasoningFallback = Math.max(defaultConfig.reasoningTimeoutSeconds, config.requestTimeoutSeconds);
    config.reasoningTimeoutSeconds = clampNumber(
      config.reasoningTimeoutSeconds ?? reasoningFallback,
      15,
      900,
      reasoningFallback
    );
    config.reasoningTimeoutSeconds = Math.max(config.reasoningTimeoutSeconds, config.requestTimeoutSeconds);
    syncReasoningTimeoutToShell(); // CODEx: Propagate timeout to standalone shell.
    config.autoInjectMemories = Boolean(config.autoInjectMemories);
    config.openRouterPolicy = (config.openRouterPolicy ?? '').trim();
  } catch (error) {
    console.error('Failed to load config:', error);
  }
}

function loadPinnedMessages() {
  try {
    const stored = storageGetItem(PINNED_STORAGE_KEY);
    if (!stored) {
      pinnedMessageIds = new Set();
      return;
    }
    const parsed = JSON.parse(stored);
    if (Array.isArray(parsed)) {
      pinnedMessageIds = new Set(parsed.map((value) => normalizeMessageId(value)));
    }
  } catch (error) {
    console.error('Failed to load pinned messages:', error);
    pinnedMessageIds = new Set();
  }
}

function loadRetrievalMetrics() {
  try {
    const stored = storageGetItem(RETRIEVAL_STORAGE_KEY);
    if (!stored) {
      retrievalMetrics = {
        A: { total: 0, lastCount: 0, lastAt: null },
        B: { total: 0, lastCount: 0, lastAt: null }
      };
      return;
    }
    const parsed = JSON.parse(stored);
    const next = { A: retrievalMetrics.A, B: retrievalMetrics.B };
    if (parsed?.A) {
      next.A = {
        total: Number(parsed.A.total) || 0,
        lastCount: Number(parsed.A.lastCount) || 0,
        lastAt: parsed.A.lastAt ? Number(parsed.A.lastAt) : null
      };
    }
    if (parsed?.B) {
      next.B = {
        total: Number(parsed.B.total) || 0,
        lastCount: Number(parsed.B.lastCount) || 0,
        lastAt: parsed.B.lastAt ? Number(parsed.B.lastAt) : null
      };
    }
    retrievalMetrics = next;
  } catch (error) {
    console.error('Failed to load retrieval metrics:', error);
    retrievalMetrics = {
      A: { total: 0, lastCount: 0, lastAt: null },
      B: { total: 0, lastCount: 0, lastAt: null }
    };
  }
  updateRetrievalStats();
}

function persistRetrievalMetrics() {
  try {
    const payload = {
      A: retrievalMetrics.A,
      B: retrievalMetrics.B
    };
    storageSetItem(RETRIEVAL_STORAGE_KEY, JSON.stringify(payload));
  } catch (error) {
    console.error('Failed to persist retrieval metrics:', error);
  }
}

function registerRetrieval(agent, count) {
  const key = agent === 'B' ? 'B' : 'A';
  const bucket = retrievalMetrics[key] ?? { total: 0, lastCount: 0, lastAt: null };
  const sanitizedCount = Math.max(0, Number(count) || 0);
  bucket.total = Math.max(0, Number(bucket.total) || 0) + sanitizedCount;
  bucket.lastCount = sanitizedCount;
  bucket.lastAt = Date.now();
  retrievalMetrics = {
    ...retrievalMetrics,
    [key]: { ...bucket }
  };
  persistRetrievalMetrics();
  updateRetrievalStats();
}

function updateRetrievalStats() {
  if (elements.modelARetrievals) {
    elements.modelARetrievals.textContent = formatRetrievalBucket(retrievalMetrics.A);
  }
  if (elements.modelBRetrievals) {
    elements.modelBRetrievals.textContent = formatRetrievalBucket(retrievalMetrics.B);
  }
}

function formatRetrievalBucket(bucket = { total: 0, lastCount: 0, lastAt: null }) {
  const total = Math.max(0, Number(bucket.total) || 0);
  const recentCount = Math.max(0, Number(bucket.lastCount) || 0);
  const lastAt = bucket.lastAt ? Number(bucket.lastAt) : null;
  if (!lastAt) {
    return `${total} total`;
  }
  const relative = formatRelativeTime(lastAt);
  return `${total} total • last: ${recentCount} (${relative})`;
}

function getRagSessionKey(mode) {
  return `${RAG_SESSION_KEY_PREFIX}-${mode}`;
}

function createRagSessionId(mode) {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  return `${mode}-session-${timestamp}`;
}

function ensureRagSessionId(mode) {
  const key = getRagSessionKey(mode);
  let sessionId = storageGetItem(key);
  if (!sessionId) {
    sessionId = createRagSessionId(mode);
    storageSetItem(key, sessionId);
  }
  if (mode === MODE_CHAT) {
    activeChatSessionId = sessionId;
  } else if (mode === MODE_ARENA) {
    activeArenaSessionId = sessionId;
  }
  return sessionId;
}

function rotateRagSession(mode) {
  const key = getRagSessionKey(mode);
  const sessionId = createRagSessionId(mode);
  storageSetItem(key, sessionId);
  if (mode === MODE_CHAT) {
    activeChatSessionId = sessionId;
    chatCheckpointBuffer = [];
  } else if (mode === MODE_ARENA) {
    activeArenaSessionId = sessionId;
    arenaCheckpointBuffer = [];
  }
  return sessionId;
}

function populateProviderOptions(select, { includeInherit = false } = {}) {
  if (!select) return;
  select.innerHTML = '';
  if (includeInherit) {
    const inheritOption = document.createElement('option');
    inheritOption.value = 'inherit';
    inheritOption.textContent = 'Share Model A settings';
    select.appendChild(inheritOption);
  }
  for (const preset of providerPresets) {
    const option = document.createElement('option');
    option.value = preset.id;
    option.textContent = preset.label;
    select.appendChild(option);
  }
}

function populateProviderSelect() {
  populateProviderOptions(elements.providerSelect);
}

function populateArenaProviderSelects() {
  populateProviderOptions(elements.agentBProviderSelect, { includeInherit: true });
}

function applyProviderPreset(presetId, { silent = false } = {}) {
  const preset = providerPresetMap.get(presetId) ?? providerPresetMap.get('custom');
  config.providerPreset = preset.id;
  if (preset.endpoint) {
    config.endpoint = preset.endpoint;
  }
  if (preset.model) {
    config.model = preset.model;
  }
  updateConfigInputs();
  saveConfig();
  updateModelConnectionStatus();
  if (!silent && preset.id !== 'custom') {
    addSystemMessage(`Loaded ${preset.label} defaults. Double-check your API key and click "Save model settings" to confirm.`);
  }
}

function updateProviderNotes() {
  if (!elements.providerNotes) return;
  const preset = providerPresetMap.get(config.providerPreset) ?? providerPresetMap.get('custom');
  const parts = [preset.description];
  if (preset.requiresKey) {
    parts.push('API key required. Paste it below before saving.');
  }
  if (preset.id === 'openrouter') {
    parts.push('Set the data policy below to match your OpenRouter privacy settings.');
  }
  elements.providerNotes.textContent = parts.filter(Boolean).join(' ');
}

function applyOpenRouterPolicyVisibility() {
  const field = elements.openRouterPolicyField;
  const input = elements.openRouterPolicyInput;
  if (!field) return;
  const primaryPreset = config.providerPreset ?? 'custom';
  const arenaPreset = config.agentBProviderPreset ?? 'inherit';
  const effectiveArenaPreset = arenaPreset === 'inherit' ? primaryPreset : arenaPreset;
  const isOpenRouter = primaryPreset === 'openrouter' || effectiveArenaPreset === 'openrouter';
  field.hidden = !isOpenRouter;
  field.setAttribute('aria-hidden', String(!isOpenRouter));
  if (input) {
    input.disabled = !isOpenRouter;
  }
}

function applyArenaProviderPreset(agent, presetId, { silent = false } = {}) {
  if (agent === 'A') {
    return;
  }

  const prefix = 'agentB';
  if (!presetId || presetId === 'inherit') {
    config[`${prefix}ProviderPreset`] = 'inherit';
    updateAgentConnectionInputs(agent);
    applyOpenRouterPolicyVisibility();
    saveConfig();
    updateModelConnectionStatus();
    if (!silent) {
      addSystemMessage(`${getAgentDisplayName(agent)} will share the Model A settings.`);
    }
    return;
  }

  const preset = providerPresetMap.get(presetId) ?? providerPresetMap.get('custom');
  config[`${prefix}ProviderPreset`] = preset.id;
  if (preset.endpoint) {
    config[`${prefix}Endpoint`] = preset.endpoint;
  }
  if (preset.model) {
    config[`${prefix}Model`] = preset.model;
  }
  updateAgentConnectionInputs(agent);
  applyOpenRouterPolicyVisibility();
  saveConfig();
  updateModelConnectionStatus();
  if (!silent && preset.id !== 'custom') {
    addSystemMessage(
      `${getAgentDisplayName(agent)} loaded ${preset.label} defaults. Provide credentials if required and click "Save model settings".`
    );
  }
}

function getAgentDisplayName(agent) {
  if (agent === 'A') {
    return config.agentAName || 'SAM-A';
  }
  if (agent === 'B') {
    return config.agentBName || 'SAM-B';
  }
  if (agent === 'system') {
    return 'System';
  }
  if (typeof agent === 'string' && agent.trim()) {
    return agent;
  }
  return 'System';
}

function getAgentPersona(agent) {
  if (agent === 'A') {
    const primary = config.agentAPrompt?.trim();
    if (primary) {
      return primary;
    }
    return config.systemPrompt?.trim() ?? '';
  }
  return config.agentBPrompt?.trim() ?? '';
}

function saveConfig() {
  storageSetItem('sam-config', JSON.stringify(config));
}

function updateConfigInputs() {
  if (elements.memorySlider) {
    elements.memorySlider.min = String(MIN_MEMORY_LIMIT_MB);
    elements.memorySlider.max = String(MAX_MEMORY_LIMIT_MB);
    elements.memorySlider.step = String(MEMORY_LIMIT_STEP_MB);
    elements.memorySlider.value = config.memoryLimitMB;
  }
  elements.memorySliderValue.innerHTML = `${config.memoryLimitMB}&nbsp;MB`;
  elements.retrievalCount.value = config.retrievalCount;
  if (elements.autoInjectMemories) {
    elements.autoInjectMemories.checked = Boolean(config.autoInjectMemories);
  }
  if (elements.reasoningModeToggle) {
    elements.reasoningModeToggle.checked = Boolean(config.reasoningModeEnabled);
  }
  if (elements.embeddingRetrievalToggle) {
    elements.embeddingRetrievalToggle.checked = isEmbeddingRetrievalEnabled(); // CODEx: Reflect embedding toggle state in UI.
  }
  refreshEmbeddingStatusIndicator(); // CODEx: Update embedding status banner after syncing configuration.
  elements.contextTurns.value = config.contextTurns;
  if (elements.providerSelect) {
    elements.providerSelect.value = config.providerPreset ?? defaultConfig.providerPreset;
  }
  elements.endpointInput.value = config.endpoint;
  elements.modelInput.value = config.model;
  elements.apiKeyInput.value = config.apiKey;
  if (elements.embeddingModelInput) {
    elements.embeddingModelInput.value = config.embeddingModel ?? defaultConfig.embeddingModel;
  }
  if (elements.embeddingEndpointInput) {
    elements.embeddingEndpointInput.value = config.embeddingEndpoint ?? '';
  }
  if (elements.embeddingApiKeyInput) {
    elements.embeddingApiKeyInput.value = config.embeddingApiKey ?? '';
  }
  if (elements.embeddingProviderSelect) {
    elements.embeddingProviderSelect.value = config.embeddingProviderPreference ?? defaultConfig.embeddingProviderPreference; // CODEx: Reflect provider preference dropdown.
  }
  if (elements.embeddingContextLength) {
    elements.embeddingContextLength.value = String(config.embeddingContextLength ?? defaultConfig.embeddingContextLength); // CODEx: Sync embedding context length input.
  }
  if (elements.agentAEmbeddingProviderSelect) {
    elements.agentAEmbeddingProviderSelect.value = config.agentAEmbeddingProvider ?? defaultConfig.agentAEmbeddingProvider; // CODEx: Reflect SAM-A embedding provider.
  }
  if (elements.agentAEmbeddingModelInput) {
    elements.agentAEmbeddingModelInput.value = config.agentAEmbeddingModel ?? defaultConfig.agentAEmbeddingModel; // CODEx: Sync SAM-A embedding model text.
  }
  if (elements.agentAEmbeddingEndpointInput) {
    elements.agentAEmbeddingEndpointInput.value = config.agentAEmbeddingEndpoint ?? ''; // CODEx: Populate SAM-A endpoint override.
  }
  if (elements.agentAEmbeddingApiKeyInput) {
    elements.agentAEmbeddingApiKeyInput.value = config.agentAEmbeddingApiKey ?? ''; // CODEx: Populate SAM-A embedding key.
  }
  if (elements.agentAEmbeddingContextInput) {
    elements.agentAEmbeddingContextInput.value = config.agentAEmbeddingContextLength ?? defaultConfig.agentAEmbeddingContextLength; // CODEx: Reflect SAM-A context length.
  }
  if (elements.agentAEmbeddingStatus) {
    elements.agentAEmbeddingStatus.textContent = ''; // CODEx: Clear SAM-A test status on load.
  }
  if (elements.agentBEmbeddingProviderSelect) {
    elements.agentBEmbeddingProviderSelect.value = config.agentBEmbeddingProvider ?? defaultConfig.agentBEmbeddingProvider; // CODEx: Reflect SAM-B embedding provider.
  }
  if (elements.agentBEmbeddingModelInput) {
    elements.agentBEmbeddingModelInput.value = config.agentBEmbeddingModel ?? defaultConfig.agentBEmbeddingModel; // CODEx: Sync SAM-B embedding model text.
  }
  if (elements.agentBEmbeddingEndpointInput) {
    elements.agentBEmbeddingEndpointInput.value = config.agentBEmbeddingEndpoint ?? ''; // CODEx: Populate SAM-B endpoint override.
  }
  if (elements.agentBEmbeddingApiKeyInput) {
    elements.agentBEmbeddingApiKeyInput.value = config.agentBEmbeddingApiKey ?? ''; // CODEx: Populate SAM-B embedding key.
  }
  if (elements.agentBEmbeddingContextInput) {
    elements.agentBEmbeddingContextInput.value = config.agentBEmbeddingContextLength ?? defaultConfig.agentBEmbeddingContextLength; // CODEx: Reflect SAM-B context length.
  }
  if (elements.agentBEmbeddingStatus) {
    elements.agentBEmbeddingStatus.textContent = ''; // CODEx: Clear SAM-B test status on load.
  }
  if (elements.openRouterPolicyInput) {
    elements.openRouterPolicyInput.value = config.openRouterPolicy ?? '';
  }
  if (elements.requestTimeout) {
    elements.requestTimeout.value = config.requestTimeoutSeconds;
  }
  if (elements.reasoningTimeout) {
    elements.reasoningTimeout.value = config.reasoningTimeoutSeconds;
  }
  elements.systemPromptInput.value = config.systemPrompt;
  const agentATemperature = config.agentATemperature ?? config.temperature ?? defaultConfig.agentATemperature; // CODEx: Resolve Agent A temperature.
  if (elements.temperatureInput) { // CODEx: Sync Agent A temperature slider.
    elements.temperatureInput.value = String(agentATemperature);
  }
  if (elements.temperatureValueLabel) { // CODEx: Update Agent A temperature label.
    elements.temperatureValueLabel.textContent = Number(agentATemperature).toFixed(2);
  }
  const agentAMaxTokens = config.agentAMaxResponseTokens ?? config.maxResponseTokens ?? defaultConfig.agentAMaxResponseTokens; // CODEx: Resolve Agent A max tokens.
  if (elements.maxTokensInput) {
    elements.maxTokensInput.value = String(agentAMaxTokens);
    updateMaxTokensCeiling();
  }
  const agentBTemperature = config.agentBTemperature ?? agentATemperature; // CODEx: Resolve Agent B temperature fallback.
  if (elements.agentBTemperatureInput) { // CODEx: Sync Agent B temperature slider.
    elements.agentBTemperatureInput.value = String(agentBTemperature);
  }
  if (elements.agentBTemperatureLabel) { // CODEx: Update Agent B temperature label.
    elements.agentBTemperatureLabel.textContent = Number(agentBTemperature).toFixed(2);
  }
  const agentBMaxTokens = config.agentBMaxResponseTokens ?? config.maxResponseTokens ?? defaultConfig.agentBMaxResponseTokens; // CODEx: Resolve Agent B max tokens.
  if (elements.agentBMaxTokensInput) { // CODEx: Sync Agent B max tokens input.
    elements.agentBMaxTokensInput.value = String(agentBMaxTokens);
  }
  updateAgentConnectionInputs('B');
  if (elements.dualSeedInput) {
    elements.dualSeedInput.value = config.dualSeed ?? DEFAULT_DUAL_SEED;
  }
  const limitValue = config.dualTurnLimit > 0 ? String(config.dualTurnLimit) : '';
  elements.dualTurnLimit.value = limitValue;
  if (elements.dualTurnDelay) {
    elements.dualTurnDelay.value = String(config.dualTurnDelaySeconds ?? defaultConfig.dualTurnDelaySeconds);
  }
  elements.dualAutoContinue.checked = config.dualAutoContinue;
  if (elements.reflexToggle) {
    elements.reflexToggle.checked = Boolean(config.reflexEnabled);
  }
  if (elements.reflexInterval) {
    elements.reflexInterval.value = String(config.reflexInterval ?? defaultConfig.reflexInterval);
  }
  if (elements.entropyWindow) {
    elements.entropyWindow.value = String(config.entropyWindow ?? defaultConfig.entropyWindow);
  }
  applyAgentBEnabledState();
  elements.agentAName.value = config.agentAName;
  elements.agentBName.value = config.agentBName;
  elements.agentBPrompt.value = config.agentBPrompt;
  if (elements.ttsPresetSelect) {
    elements.ttsPresetSelect.value = config.ttsPreset;
  }
  if (elements.ttsServerUrl) {
    elements.ttsServerUrl.value = config.ttsServerUrl ?? '';
  }
  if (elements.ttsVoiceId) {
    elements.ttsVoiceId.value = config.ttsVoiceId ?? '';
  }
  if (elements.ttsApiKeyInput) {
    elements.ttsApiKeyInput.value = config.ttsApiKey ?? '';
  }
  if (elements.ttsVolume) {
    elements.ttsVolume.value = String(config.ttsVolume ?? defaultConfig.ttsVolume);
  }
  if (elements.debugToggle) {
    elements.debugToggle.checked = Boolean(config.debugEnabled);
  }
  updateProviderNotes();
  applyOpenRouterPolicyVisibility();
  updateTtsPresetDetails();
  updateTtsVolumeLabel();
  if (elements.backgroundUrlInput) {
    elements.backgroundUrlInput.value =
      config.backgroundSource === 'url' ? config.backgroundUrl ?? '' : '';
  }
  updateBackgroundStatus();
}

function updateAgentConnectionInputs(agent) {
  if (agent === 'A') {
    updateArenaProviderNotes('A', getAgentConnection('A'));
    return;
  }

  const prefix = 'agentB';
  const providerSelect = elements.agentBProviderSelect;
  const endpointInput = elements.agentBEndpoint;
  const modelInput = elements.agentBModel;
  const apiKeyInput = elements.agentBApiKey;
  const providerKey = config[`${prefix}ProviderPreset`] ?? 'inherit';
  const connection = getAgentConnection(agent);

  if (providerSelect) {
    providerSelect.value = providerKey;
  }

  if (!config.agentBEnabled) {
    if (providerSelect) providerSelect.disabled = true;
    if (endpointInput) endpointInput.disabled = true;
    if (modelInput) modelInput.disabled = true;
    if (apiKeyInput) apiKeyInput.disabled = true;
    return;
  }

  if (endpointInput) {
    if (!endpointInput.dataset.placeholder) {
      endpointInput.dataset.placeholder = endpointInput.placeholder;
    }
    endpointInput.value = config[`${prefix}Endpoint`] ?? '';
    endpointInput.disabled = connection.inherits;
    endpointInput.placeholder = connection.inherits
      ? connection.endpoint || config.endpoint || 'Using main connection'
      : endpointInput.dataset.placeholder;
  }

  if (modelInput) {
    if (!modelInput.dataset.placeholder) {
      modelInput.dataset.placeholder = modelInput.placeholder;
    }
    modelInput.value = config[`${prefix}Model`] ?? '';
    modelInput.disabled = connection.inherits;
    modelInput.placeholder = connection.inherits
      ? connection.model || config.model || 'Using main connection'
      : modelInput.dataset.placeholder;
  }

  if (apiKeyInput) {
    if (!apiKeyInput.dataset.placeholder) {
      apiKeyInput.dataset.placeholder = apiKeyInput.placeholder;
    }
    apiKeyInput.value = config[`${prefix}ApiKey`] ?? '';
    apiKeyInput.disabled = connection.inherits;
    apiKeyInput.placeholder = connection.inherits
      ? config.apiKey
        ? 'Using main connection key'
        : 'Using main connection (optional)'
      : apiKeyInput.dataset.placeholder;
  }

  const embeddingControls = [
    elements.agentBEmbeddingProviderSelect,
    elements.agentBEmbeddingModelInput,
    elements.agentBEmbeddingEndpointInput,
    elements.agentBEmbeddingApiKeyInput,
    elements.agentBEmbeddingContextInput,
    elements.agentBEmbeddingTestButton
  ];
  for (const node of embeddingControls) {
    if (!node) continue;
    node.disabled = connection.inherits; // CODEx: Lock SAM-B embedding controls when inheriting Model A settings.
  }

  updateArenaProviderNotes(agent, connection);
}

function applyAgentBEnabledState() {
  const enabled = config.agentBEnabled !== false;
  if (elements.agentBEnabled) {
    elements.agentBEnabled.checked = enabled;
  }
  const toggledFields = [
    'agentBProviderSelect',
    'agentBEndpoint',
    'agentBModel',
    'agentBApiKey',
    'agentBName',
    'agentBPrompt',
    'agentBTemperatureInput',
    'agentBMaxTokensInput',
    'agentBEmbeddingProviderSelect', // CODEx: Disable SAM-B embedding provider when Model B is off.
    'agentBEmbeddingModelInput', // CODEx: Disable SAM-B embedding model input when Model B is off.
    'agentBEmbeddingEndpointInput', // CODEx: Disable SAM-B endpoint override when Model B is off.
    'agentBEmbeddingApiKeyInput', // CODEx: Disable SAM-B embedding key when Model B is off.
    'agentBEmbeddingContextInput', // CODEx: Disable SAM-B context length when Model B is off.
    'agentBEmbeddingTestButton' // CODEx: Disable SAM-B test trigger when Model B is off.
  ];
  for (const key of toggledFields) {
    const node = elements[key];
    if (node) {
      if (!enabled) {
        node.disabled = true;
      } else {
        node.disabled = false;
      }
    }
  }
  if (elements.agentBBody) {
    elements.agentBBody.setAttribute('aria-disabled', String(!enabled)); // CODEx: Mark Agent B body disabled state.
  }
  if (enabled) {
    updateAgentConnectionInputs('B');
  }
  updateProcessState(
    'modelB',
    enabled
      ? { status: 'Idle', detail: 'Awaiting arena start.' }
      : { status: 'Disabled', detail: 'Enable Model B to run dual debates.' }
  );
}

function updateArenaProviderNotes(agent, connection = getAgentConnection(agent)) {
  if (agent === 'A') return;
  const notesElement = elements.agentBProviderNotes;
  if (!notesElement) return;
  if (connection.inherits) {
    const baseLabel = connection.providerLabel || 'Custom';
    notesElement.textContent = `Sharing Model A settings (${baseLabel}).`;
    return;
  }
  const parts = [connection.providerLabel];
  if (connection.presetDescription) {
    parts.push(connection.presetDescription);
  }
  if (connection.requiresKey) {
    parts.push('API key required. Paste it below before saving.');
  }
  notesElement.textContent = parts.filter(Boolean).join(' ');
}

function getAgentConnection(agent) {
  const basePreset = providerPresetMap.get(config.providerPreset) ?? providerPresetMap.get('custom');

  if (agent === 'A') {
    return {
      endpoint: config.endpoint,
      model: config.model,
      apiKey: config.apiKey,
      temperature: config.agentATemperature ?? config.temperature ?? defaultConfig.agentATemperature,
      maxTokens: config.agentAMaxResponseTokens ?? config.maxResponseTokens ?? defaultConfig.agentAMaxResponseTokens,
      providerPreset: basePreset?.id ?? 'custom',
      providerLabel: basePreset ? `${basePreset.label} (main)` : 'Custom',
      presetDescription: basePreset?.description ?? '',
      requiresKey: Boolean(basePreset?.requiresKey),
      inherits: false,
      agentPresetId: basePreset?.id ?? 'custom',
      openRouterPolicy: config.openRouterPolicy
    };
  }

  const agentPresetId = config.agentBProviderPreset ?? 'inherit';
  const inherits = agentPresetId === 'inherit';
  const preset = inherits
    ? basePreset
    : providerPresetMap.get(agentPresetId) ?? providerPresetMap.get('custom');

  const endpoint = inherits ? config.endpoint : config.agentBEndpoint || preset?.endpoint || '';
  const model = inherits ? config.model : config.agentBModel || preset?.model || '';
  const apiKey = inherits ? config.apiKey : config.agentBApiKey || '';
  const aTemperature = config.agentATemperature ?? config.temperature ?? defaultConfig.agentATemperature;
  const aMaxTokens = config.agentAMaxResponseTokens ?? config.maxResponseTokens ?? defaultConfig.agentAMaxResponseTokens;
  const bTemperature = config.agentBTemperature ?? aTemperature;
  const bMaxTokens = config.agentBMaxResponseTokens ?? aMaxTokens;

  return {
    endpoint,
    model,
    apiKey,
    temperature: bTemperature,
    maxTokens: bMaxTokens,
    providerPreset: preset?.id ?? 'custom',
    providerLabel: inherits && basePreset ? `${basePreset.label} (main)` : preset?.label ?? 'Custom',
    presetDescription: preset?.description ?? '',
    requiresKey: Boolean(preset?.requiresKey),
    inherits,
    agentPresetId,
    openRouterPolicy: config.openRouterPolicy
  };
}

function formatArenaConnectionDiagnostic(agent, connection = getAgentConnection(agent)) {
  const name = getAgentDisplayName(agent);
  if (agent === 'B' && !config.agentBEnabled) {
    return `${name}: Disabled (enable Model B to run dual debates).`;
  }
  if (!connection.endpoint || !connection.model) {
    const missingScope = connection.inherits ? 'main connection' : 'arena connection';
    return `${name}: Not configured (missing endpoint/model on ${missingScope}).`;
  }
  const providerNote = connection.inherits ? 'sharing main connection' : connection.providerLabel;
  const suffix = providerNote ? ` • ${providerNote}` : '';
  return `${name}: ${connection.model} @ ${connection.endpoint}${suffix}`;
}

function toggleOptionsPanel(forceState) {
  const shouldOpen = typeof forceState === 'boolean' ? forceState : !document.body.classList.contains('options-open');
  if (shouldOpen) {
    openOptionsPanel();
  } else {
    closeOptionsPanel();
  }
}

function handleBackgroundUrlChange(rawValue) {
  const value = (rawValue || '').trim();
  if (value) {
    config.backgroundSource = 'url';
    config.backgroundUrl = value;
    config.backgroundImage = '';
    saveConfig();
    applyBackgroundFromConfig();
    updateBackgroundStatus(`Using background image from ${value}`);
  } else if (config.backgroundSource === 'url') {
    config.backgroundUrl = '';
    config.backgroundSource = config.backgroundImage ? 'upload' : 'default';
    saveConfig();
    applyBackgroundFromConfig();
  }
  if (elements.backgroundUrlInput && elements.backgroundUrlInput.value !== value) {
    elements.backgroundUrlInput.value = value;
  }
}

function handleBackgroundFileSelection(file) {
  if (!file) {
    return;
  }
  const maxBytes = 8 * MB;
  if (file.size > maxBytes) {
    updateBackgroundStatus(`Image is too large (${formatMegabytes(file.size)}). Choose a file under 8 MB.`);
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    const result = reader.result;
    if (typeof result === 'string') {
      config.backgroundSource = 'upload';
      config.backgroundImage = result;
      config.backgroundUrl = '';
      saveConfig();
      applyBackgroundFromConfig();
      updateBackgroundStatus(`Custom backdrop applied from ${file.name}`);
    }
  };
  reader.onerror = () => {
    console.error('Failed to read background image', reader.error);
    updateBackgroundStatus('Failed to load the selected image. Please try another file.');
  };
  reader.readAsDataURL(file);
}

function clearBackgroundImage() {
  config.backgroundSource = 'default';
  config.backgroundImage = '';
  config.backgroundUrl = '';
  saveConfig();
  applyBackgroundFromConfig();
  updateBackgroundStatus('Default gradient background active.');
  if (elements.backgroundUrlInput) {
    elements.backgroundUrlInput.value = '';
  }
}

function applyBackgroundFromConfig() {
  const body = document.body;
  const root = document.documentElement;
  if (!body) return;
  const source = config.backgroundSource ?? 'default';
  let backgroundValue = DEFAULT_BACKGROUND;
  let hasCustom = false;
  if (source === 'upload' && config.backgroundImage) {
    backgroundValue = `${CUSTOM_BACKGROUND_OVERLAY}, url(${config.backgroundImage}), ${DEFAULT_BACKGROUND}`;
    hasCustom = true;
  } else if (source === 'url' && config.backgroundUrl) {
    backgroundValue = `${CUSTOM_BACKGROUND_OVERLAY}, url(${config.backgroundUrl}), ${DEFAULT_BACKGROUND}`;
    hasCustom = true;
  }
  if (root) {
    root.style.setProperty('--app-bg', backgroundValue);
  }
  body.style.setProperty('--app-bg', backgroundValue);
  body.dataset.hasCustomBg = hasCustom ? 'true' : 'false';
  updateBackgroundStatus();
}

function updateBackgroundStatus(explicitMessage) {
  if (!elements.backgroundStatus) return;
  if (explicitMessage) {
    elements.backgroundStatus.textContent = explicitMessage;
    return;
  }
  const source = config.backgroundSource ?? 'default';
  if (source === 'upload' && config.backgroundImage) {
    elements.backgroundStatus.textContent = 'Custom backdrop active from uploaded image.';
  } else if (source === 'url' && config.backgroundUrl) {
    elements.backgroundStatus.textContent = `Using background URL: ${config.backgroundUrl}`;
  } else {
    elements.backgroundStatus.textContent = 'Default gradient background active.';
  }
}

function openOptionsPanel() {
  document.body.classList.add('options-open');
  if (elements.optionsHandle) {
    elements.optionsHandle.setAttribute('aria-expanded', 'true');
  }
  if (elements.optionsBackdrop) {
    elements.optionsBackdrop.hidden = false;
  }
  if (elements.optionsDrawer) {
    elements.optionsDrawer.removeAttribute('aria-hidden');
    elements.optionsDrawer.removeAttribute('inert');
    elements.optionsDrawer.focus({ preventScroll: false });
  }
}

function closeOptionsPanel() {
  document.body.classList.remove('options-open');
  if (elements.optionsHandle) {
    elements.optionsHandle.setAttribute('aria-expanded', 'false');
  }
  if (elements.optionsBackdrop) {
    elements.optionsBackdrop.hidden = true;
  }
  if (elements.optionsDrawer) {
    elements.optionsDrawer.setAttribute('aria-hidden', 'true');
    elements.optionsDrawer.setAttribute('inert', '');
  }
}

function bindEvents() {
  // Store event listeners for cleanup
  const eventListeners = [];

  const addListener = (element, event, handler) => {
    if (element) {
      element.addEventListener(event, handler);
      eventListeners.push({ element, event, handler });
    }
  };

  if (elements.agentACard && elements.toggleAgentA && elements.agentABody) { // CODEx: Initialize Agent A accordion state.
    const collapsed = elements.agentACard.getAttribute('data-collapsed') === 'true'; // CODEx: Determine default collapse value.
    updateCardSectionState(elements.agentACard, elements.toggleAgentA, elements.agentABody, collapsed); // CODEx: Sync visuals.
    addListener(elements.toggleAgentA, 'click', () => { // CODEx: Toggle Agent A card on demand.
      toggleCardSection(elements.agentACard, elements.toggleAgentA, elements.agentABody); // CODEx: Apply flip state.
    });
  }

  if (elements.agentBCard && elements.toggleAgentB && elements.agentBBody) { // CODEx: Initialize Agent B accordion state.
    const collapsed = elements.agentBCard.getAttribute('data-collapsed') === 'true'; // CODEx: Determine default collapse value.
    updateCardSectionState(elements.agentBCard, elements.toggleAgentB, elements.agentBBody, collapsed); // CODEx: Sync visuals.
    addListener(elements.toggleAgentB, 'click', () => { // CODEx: Toggle Agent B card on demand.
      toggleCardSection(elements.agentBCard, elements.toggleAgentB, elements.agentBBody); // CODEx: Apply flip state.
    });
  }

  addListener(elements.enterChatButton, 'click', () => {
    setActiveMode(MODE_CHAT);
  });

  addListener(elements.enterArenaButton, 'click', () => {
    setActiveMode(MODE_ARENA);
  });

  addListener(elements.switchToChatButton, 'click', () => {
    setActiveMode(MODE_CHAT);
  });

  addListener(elements.switchToArenaButton, 'click', () => {
    setActiveMode(MODE_ARENA);
  });

  addListener(elements.sendButton, 'click', () => {
    void handleUserMessage();
  });

  addListener(elements.messageInput, 'keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void handleUserMessage();
    }
  });

  addListener(elements.retrieveButton, 'click', () => {
    void promoteRelevantMemories();
  });

  addListener(elements.clearChatButton, 'click', clearChatWindow);

  addListener(elements.memorySlider, 'input', (event) => {
    const rawValue = Number.parseInt(event.target.value, 10);
    const value = clampNumber(rawValue, MIN_MEMORY_LIMIT_MB, MAX_MEMORY_LIMIT_MB, config.memoryLimitMB);
    if (String(value) !== event.target.value) {
      event.target.value = String(value);
    }
    config.memoryLimitMB = value;
    elements.memorySliderValue.innerHTML = `${value}&nbsp;MB`;
    saveConfig();
    trimFloatingMemory();
  });

  addListener(elements.retrievalCount, 'change', (event) => {
    config.retrievalCount = clampNumber(event.target.value, 0, 200, config.retrievalCount);
    elements.retrievalCount.value = config.retrievalCount;
    saveConfig();
  });

  addListener(elements.retrievalCount, 'input', (event) => {
    config.retrievalCount = clampNumber(event.target.value, 0, 200, config.retrievalCount);
    saveConfig();
  });

  addListener(elements.reasoningModeToggle, 'change', (event) => {
    config.reasoningModeEnabled = event.target.checked;
    saveConfig();
    updateMemoryStatus();
  });

  addListener(elements.autoInjectMemories, 'change', (event) => {
    config.autoInjectMemories = event.target.checked;
    saveConfig();
    updateRagStatusDisplay();
  });
  // CODEx: Separate memory injection handling from embedding toggle logic for clarity.
  addListener(elements.embeddingRetrievalToggle, 'change', (event) => {
    config.useEmbeddingRetrieval = event.target.checked; // CODEx: Persist embedding toggle state on interaction.
    saveConfig(); // CODEx: Store updated configuration immediately.
    refreshEmbeddingStatusIndicator(); // CODEx: Reflect new embedding availability in the UI.
  });

  addListener(elements.contextTurns, 'change', (event) => {
    config.contextTurns = clampNumber(event.target.value, 2, 40, config.contextTurns);
    elements.contextTurns.value = config.contextTurns;
    saveConfig();
  });

  addListener(elements.providerSelect, 'change', (event) => {
    applyProviderPreset(event.target.value, { silent: false });
  });

  addListener(elements.agentBProviderSelect, 'change', (event) => {
    applyArenaProviderPreset('B', event.target.value);
  });

  const autosaveTargets = document.querySelectorAll(
    '#agentABody input, #agentABody select, #agentABody textarea, #agentBBody input, #agentBBody select, #agentBBody textarea'
  ); // CODEx: Aggregate Agent card form fields for blur autosave.
  autosaveTargets.forEach((node) => {
    addListener(node, 'blur', () => {
      saveConfig();
      if (elements.saveStatus) {
        elements.saveStatus.textContent = 'Saved.';
        window.clearTimeout(elements.saveStatus._clearTimer);
        elements.saveStatus._clearTimer = window.setTimeout(() => {
          elements.saveStatus.textContent = '';
        }, 1500);
      }
    });
  });

  addListener(elements.agentBEnabled, 'change', (event) => {
    config.agentBEnabled = event.target.checked;
    saveConfig();
    applyAgentBEnabledState();
    updateAgentConnectionInputs('B');
    updateModelConnectionStatus();
  });

  addListener(elements.endpointInput, 'input', (event) => {
    config.endpoint = event.target.value.trim();
    updateAgentConnectionInputs('B');
    void resolveActiveEmbeddingProvider(true); // CODEx: Re-probe embedding provider when base endpoint changes.
  });

  addListener(elements.modelInput, 'input', (event) => {
    config.model = event.target.value.trim();
    updateAgentConnectionInputs('B');
  });

  addListener(elements.apiKeyInput, 'input', (event) => {
    config.apiKey = event.target.value;
    updateAgentConnectionInputs('B');
    void resolveActiveEmbeddingProvider(true); // CODEx: Re-evaluate embedding provider when shared API key updates.
  });

  if (elements.embeddingModelInput) {
    addListener(elements.embeddingModelInput, 'change', (event) => {
      config.embeddingModel = event.target.value.trim() || defaultConfig.embeddingModel;
      saveConfig();
      clearCachedEmbeddings();
    });
  }

  if (elements.embeddingEndpointInput) {
    addListener(elements.embeddingEndpointInput, 'input', (event) => {
      config.embeddingEndpoint = event.target.value.trim();
      saveConfig();
      clearCachedEmbeddings();
      void resolveActiveEmbeddingProvider(true); // CODEx: Refresh provider detection when embedding endpoint overrides change.
    });
  }

  if (elements.embeddingApiKeyInput) {
    addListener(elements.embeddingApiKeyInput, 'input', (event) => {
      config.embeddingApiKey = event.target.value.trim();
      saveConfig();
      clearCachedEmbeddings();
      void resolveActiveEmbeddingProvider(true); // CODEx: Trigger provider probe to reflect new embedding key.
    });
  }

  if (elements.embeddingProviderSelect) {
    addListener(elements.embeddingProviderSelect, 'change', (event) => {
      config.embeddingProviderPreference = event.target.value; // CODEx: Persist provider preference selection.
      saveConfig(); // CODEx: Retain selection across sessions.
      void resolveActiveEmbeddingProvider(true); // CODEx: Re-probe provider immediately after preference change.
    });
  }

  if (elements.embeddingContextLength) {
    const applyContextLength = (value) => {
      config.embeddingContextLength = clampNumber(value, 512, 4096, config.embeddingContextLength ?? defaultConfig.embeddingContextLength); // CODEx: Clamp context length input.
      elements.embeddingContextLength.value = String(config.embeddingContextLength);
      saveConfig();
    };
    addListener(elements.embeddingContextLength, 'change', (event) => {
      applyContextLength(event.target.value);
      clearCachedEmbeddings();
      void warmEmbeddings(); // CODEx: Re-prime embeddings after context change.
    });
    addListener(elements.embeddingContextLength, 'input', (event) => {
      applyContextLength(event.target.value);
    });
  }

  if (elements.agentAEmbeddingProviderSelect) {
    addListener(elements.agentAEmbeddingProviderSelect, 'change', (event) => {
      config.agentAEmbeddingProvider = event.target.value;
      saveConfig();
      clearCachedEmbeddings();
      void warmEmbeddings();
    });
  }

  if (elements.agentAEmbeddingModelInput) {
    addListener(elements.agentAEmbeddingModelInput, 'change', (event) => {
      const value = event.target.value.trim();
      config.agentAEmbeddingModel = value || defaultConfig.agentAEmbeddingModel;
      saveConfig();
      clearCachedEmbeddings();
      void warmEmbeddings();
    });
  }

  if (elements.agentAEmbeddingEndpointInput) {
    addListener(elements.agentAEmbeddingEndpointInput, 'input', (event) => {
      config.agentAEmbeddingEndpoint = event.target.value.trim();
      saveConfig();
      clearCachedEmbeddings();
    });
  }

  if (elements.agentAEmbeddingApiKeyInput) {
    addListener(elements.agentAEmbeddingApiKeyInput, 'input', (event) => {
      config.agentAEmbeddingApiKey = event.target.value.trim();
      saveConfig();
      clearCachedEmbeddings();
    });
  }

  if (elements.agentAEmbeddingContextInput) {
    addListener(elements.agentAEmbeddingContextInput, 'change', (event) => {
      const parsed = Number.parseInt(event.target.value, 10);
      config.agentAEmbeddingContextLength = clampNumber(parsed, 512, 4096, defaultConfig.agentAEmbeddingContextLength);
      event.target.value = config.agentAEmbeddingContextLength;
      saveConfig();
      clearCachedEmbeddings();
    });
  }

  if (elements.agentBEmbeddingProviderSelect) {
    addListener(elements.agentBEmbeddingProviderSelect, 'change', (event) => {
      config.agentBEmbeddingProvider = event.target.value;
      saveConfig();
      clearCachedEmbeddings();
      void warmEmbeddings();
    });
  }

  if (elements.agentBEmbeddingModelInput) {
    addListener(elements.agentBEmbeddingModelInput, 'change', (event) => {
      const value = event.target.value.trim();
      config.agentBEmbeddingModel = value || defaultConfig.agentBEmbeddingModel;
      saveConfig();
      clearCachedEmbeddings();
      void warmEmbeddings();
    });
  }

  if (elements.agentBEmbeddingEndpointInput) {
    addListener(elements.agentBEmbeddingEndpointInput, 'input', (event) => {
      config.agentBEmbeddingEndpoint = event.target.value.trim();
      saveConfig();
      clearCachedEmbeddings();
    });
  }

  if (elements.agentBEmbeddingApiKeyInput) {
    addListener(elements.agentBEmbeddingApiKeyInput, 'input', (event) => {
      config.agentBEmbeddingApiKey = event.target.value.trim();
      saveConfig();
      clearCachedEmbeddings();
    });
  }

  if (elements.agentBEmbeddingContextInput) {
    addListener(elements.agentBEmbeddingContextInput, 'change', (event) => {
      const parsed = Number.parseInt(event.target.value, 10);
      config.agentBEmbeddingContextLength = clampNumber(parsed, 512, 4096, defaultConfig.agentBEmbeddingContextLength);
      event.target.value = config.agentBEmbeddingContextLength;
      saveConfig();
      clearCachedEmbeddings();
    });
  }

  if (elements.agentAEmbeddingTestButton) {
    addListener(elements.agentAEmbeddingTestButton, 'click', () => {
      void runEmbeddingTest('A'); // CODEx: Execute SAM-A embedding test utility.
    });
  }

  if (elements.agentBEmbeddingTestButton) {
    addListener(elements.agentBEmbeddingTestButton, 'click', () => {
      void runEmbeddingTest('B'); // CODEx: Execute SAM-B embedding test utility.
    });
  }

  if (elements.openRouterPolicyInput) {
    addListener(elements.openRouterPolicyInput, 'input', (event) => {
      config.openRouterPolicy = event.target.value.trim();
      saveConfig();
    });
  }

  addListener(elements.requestTimeout, 'change', (event) => {
    config.requestTimeoutSeconds = clampNumber(event.target.value, 5, 600, config.requestTimeoutSeconds);
    elements.requestTimeout.value = config.requestTimeoutSeconds;
    if (config.reasoningTimeoutSeconds < config.requestTimeoutSeconds) {
      config.reasoningTimeoutSeconds = config.requestTimeoutSeconds;
      if (elements.reasoningTimeout) {
        elements.reasoningTimeout.value = config.reasoningTimeoutSeconds;
      }
      syncReasoningTimeoutToShell(); // CODEx: Keep shell timeout aligned with new baseline.
    }
    saveConfig();
  });

  addListener(elements.requestTimeout, 'input', (event) => {
    config.requestTimeoutSeconds = clampNumber(event.target.value, 5, 600, config.requestTimeoutSeconds);
    elements.requestTimeout.value = config.requestTimeoutSeconds;
    if (config.reasoningTimeoutSeconds < config.requestTimeoutSeconds) {
      config.reasoningTimeoutSeconds = config.requestTimeoutSeconds;
      if (elements.reasoningTimeout) {
        elements.reasoningTimeout.value = config.reasoningTimeoutSeconds;
      }
      syncReasoningTimeoutToShell(); // CODEx: Reflect live baseline adjustments.
    }
    saveConfig();
  });

  addListener(elements.reasoningTimeout, 'change', (event) => {
    config.reasoningTimeoutSeconds = clampNumber(event.target.value, 15, 900, config.reasoningTimeoutSeconds);
    if (config.reasoningTimeoutSeconds < config.requestTimeoutSeconds) {
      config.reasoningTimeoutSeconds = config.requestTimeoutSeconds;
    }
    elements.reasoningTimeout.value = config.reasoningTimeoutSeconds;
    syncReasoningTimeoutToShell(); // CODEx: Mirror manual changes into standalone shell.
    saveConfig();
  });

  addListener(elements.reasoningTimeout, 'input', (event) => {
    config.reasoningTimeoutSeconds = clampNumber(event.target.value, 15, 900, config.reasoningTimeoutSeconds);
    if (config.reasoningTimeoutSeconds < config.requestTimeoutSeconds) {
      config.reasoningTimeoutSeconds = config.requestTimeoutSeconds;
    }
    elements.reasoningTimeout.value = config.reasoningTimeoutSeconds;
    syncReasoningTimeoutToShell(); // CODEx: Sync live slider adjustments to shell.
    saveConfig();
  });

  addListener(elements.agentBEndpoint, 'input', (event) => {
    config.agentBEndpoint = event.target.value.trim();
    saveConfig();
  });

  addListener(elements.agentBModel, 'input', (event) => {
    config.agentBModel = event.target.value.trim();
    saveConfig();
  });

  addListener(elements.agentBApiKey, 'input', (event) => {
    config.agentBApiKey = event.target.value;
    saveConfig();
  });

  addListener(elements.systemPromptInput, 'input', (event) => {
    config.systemPrompt = event.target.value;
  });

  const applyAgentATemperature = (value) => { // CODEx: Normalize Agent A temperature updates.
    const parsed = Number.parseFloat(value);
    const clamped = clampNumber(Number.isNaN(parsed) ? config.agentATemperature ?? defaultConfig.agentATemperature : parsed, 0, 2, config.agentATemperature ?? defaultConfig.agentATemperature);
    config.agentATemperature = clamped;
    config.temperature = clamped; // CODEx: Maintain legacy temperature field for compatibility.
    if (elements.temperatureInput) {
      elements.temperatureInput.value = String(clamped);
    }
    if (elements.temperatureValueLabel) {
      elements.temperatureValueLabel.textContent = clamped.toFixed(2);
    }
    if (activeDualConnections.A) {
      activeDualConnections.A.temperature = clamped;
    }
    saveConfig();
  };

  const applyAgentBTemperature = (value) => { // CODEx: Normalize Agent B temperature updates.
    const parsed = Number.parseFloat(value);
    const base = config.agentATemperature ?? config.temperature ?? defaultConfig.agentATemperature;
    const clamped = clampNumber(Number.isNaN(parsed) ? config.agentBTemperature ?? base : parsed, 0, 2, config.agentBTemperature ?? base);
    config.agentBTemperature = clamped;
    if (elements.agentBTemperatureInput) {
      elements.agentBTemperatureInput.value = String(clamped);
    }
    if (elements.agentBTemperatureLabel) {
      elements.agentBTemperatureLabel.textContent = clamped.toFixed(2);
    }
    if (activeDualConnections.B) {
      activeDualConnections.B.temperature = clamped;
    }
    saveConfig();
  };

  addListener(elements.temperatureInput, 'input', (event) => {
    applyAgentATemperature(event.target.value);
  });

  addListener(elements.temperatureInput, 'change', (event) => {
    applyAgentATemperature(event.target.value);
  });

  if (elements.agentBTemperatureInput) {
    addListener(elements.agentBTemperatureInput, 'input', (event) => {
      applyAgentBTemperature(event.target.value);
    });
    addListener(elements.agentBTemperatureInput, 'change', (event) => {
      applyAgentBTemperature(event.target.value);
    });
  }

  addListener(elements.maxTokensInput, 'change', (event) => {
    const parsed = Number.parseInt(event.target.value, 10);
    const limit = getProviderContextLimit(config.providerPreset);
    if (Number.isNaN(parsed) || parsed <= 0) {
      const fallback = clampNumber(defaultConfig.agentAMaxResponseTokens, 16, limit, defaultConfig.agentAMaxResponseTokens);
      config.agentAMaxResponseTokens = Math.round(fallback);
    } else {
      config.agentAMaxResponseTokens = Math.round(
        clampNumber(parsed, 16, limit, config.agentAMaxResponseTokens ?? defaultConfig.agentAMaxResponseTokens)
      );
    }
    config.maxResponseTokens = config.agentAMaxResponseTokens; // CODEx: Maintain legacy max tokens mirror.
    updateMaxTokensCeiling();
    if (activeDualConnections.A) {
      activeDualConnections.A.maxTokens = config.agentAMaxResponseTokens;
    }
    if (activeDualConnections.B) {
      activeDualConnections.B.maxTokens = config.agentBMaxResponseTokens ?? config.agentAMaxResponseTokens;
    }
    saveConfig();
  });

  if (elements.agentBMaxTokensInput) {
    addListener(elements.agentBMaxTokensInput, 'change', (event) => {
      const parsed = Number.parseInt(event.target.value, 10);
      const presetId = config.agentBProviderPreset === 'inherit' || !config.agentBProviderPreset
        ? config.providerPreset
        : config.agentBProviderPreset;
      const limit = getProviderContextLimit(presetId);
      const fallback = config.agentAMaxResponseTokens ?? config.maxResponseTokens ?? defaultConfig.agentAMaxResponseTokens;
      config.agentBMaxResponseTokens = Math.round(
        clampNumber(Number.isNaN(parsed) || parsed <= 0 ? fallback : parsed, 16, limit, fallback)
      );
      if (elements.agentBMaxTokensInput) {
        elements.agentBMaxTokensInput.value = String(config.agentBMaxResponseTokens);
      }
      if (activeDualConnections.B) {
        activeDualConnections.B.maxTokens = config.agentBMaxResponseTokens;
      }
      saveConfig();
    });
  }

  addListener(elements.saveConfigButton, 'click', () => {
    saveConfig();
    updateModelConnectionStatus();
    if (document.body.classList.contains('options-open')) {
      closeOptionsPanel();
    }
    if (elements.saveStatus) {
      elements.saveStatus.textContent = 'Saved.';
      window.clearTimeout(elements.saveStatus._clearTimer);
      elements.saveStatus._clearTimer = window.setTimeout(() => {
        elements.saveStatus.textContent = '';
      }, 1500);
    }
  });

  addListener(elements.speakToggle, 'change', (event) => {
    config.autoSpeak = event.target.checked;
    saveConfig();
  });

  addListener(elements.optionsHandle, 'click', () => {
    toggleOptionsPanel();
  });

  addListener(elements.optionsBackdrop, 'click', () => {
    closeOptionsPanel();
  });

  addListener(document, 'keydown', (event) => {
    if (event.key === 'Escape') {
      if (debugPanelVisible) {
        closeDebugPanel();
        event.preventDefault();
        return;
      }
      if (document.body.classList.contains('options-open')) {
        closeOptionsPanel();
      }
    }
  });

  addListener(elements.diagnosticsButton, 'click', () => {
    void runDiagnostics();
  });

  addListener(elements.voiceButton, 'click', toggleVoiceInput);

  addListener(elements.exportButton, 'click', () => {
    exportConversation();
  });

  addListener(elements.refreshFloatingButton, 'click', () => {
    renderFloatingMemoryWorkbench();
  });

  addListener(elements.loadRagButton, 'click', () => {
    void manualRagReload();
  });

  addListener(elements.loadChunkDataButton, 'click', () => {
    void manualChunkReload();
  });

  addListener(elements.saveChatArchiveButton, 'click', () => {
    void saveChatTranscriptToArchive();
  });

  addListener(elements.archiveUnpinnedButton, 'click', () => {
    archiveUnpinnedFloatingMemory();
  });

  addListener(elements.floatingMemoryList, 'click', handleFloatingMemoryListClick);

  addListener(elements.startDualButton, 'click', () => {
    void startDualChat();
  });

  addListener(elements.nextDualButton, 'click', () => {
    void advanceDualTurn();
  });

  addListener(elements.stopDualButton, 'click', stopDualChat);

  addListener(elements.exportDualButton, 'click', exportDualTranscript);

  addListener(elements.dualSeedInput, 'input', (event) => {
    const value = event.target.value.trim();
    config.dualSeed = value || DEFAULT_DUAL_SEED;
    saveConfig();
  });

  addListener(elements.dualAutoContinue, 'change', (event) => {
    config.dualAutoContinue = event.target.checked;
    saveConfig();
    if (!config.dualAutoContinue) {
      if (autoContinueTimer) {
        clearTimeout(autoContinueTimer);
        autoContinueTimer = undefined;
      }
      clearDualCountdownTimer();
      if (isDualChatRunning && elements.dualStatus) {
        elements.dualStatus.textContent = 'Auto-continue disabled. Use Step turn to advance.';
      }
    } else if (isDualChatRunning) {
      scheduleNextDualTurn();
    }
  });

  addListener(elements.reflexToggle, 'change', (event) => {
    config.reflexEnabled = event.target.checked;
    saveConfig();
    updateReflexStatus();
  });

  addListener(elements.reflexInterval, 'change', (event) => {
    const value = Number.parseInt(event.target.value, 10);
    config.reflexInterval = clampNumber(value, 2, 50, defaultConfig.reflexInterval);
    event.target.value = String(config.reflexInterval);
    saveConfig();
  });

  addListener(elements.entropyWindow, 'change', (event) => {
    const value = Number.parseInt(event.target.value, 10);
    config.entropyWindow = clampNumber(value, 1, 25, defaultConfig.entropyWindow);
    event.target.value = String(config.entropyWindow);
    saveConfig();
    updateEntropyMeter();
  });

  addListener(elements.triggerReflexButton, 'click', () => {
    const lastSpeaker = dualChatHistory[dualChatHistory.length - 1]?.speaker;
    let target = lastSpeaker || 'A';
    if (target === 'B' && !config.agentBEnabled) {
      target = 'A';
    }
    void runReflexSummary(target, { forced: true });
  });

  addListener(elements.dualTurnLimit, 'change', (event) => {
    const raw = event.target.value.trim();
    if (!raw) {
      config.dualTurnLimit = 0;
    } else {
      const parsed = Number.parseInt(raw, 10);
      config.dualTurnLimit = Number.isNaN(parsed) || parsed < 0 ? 0 : parsed;
    }
    elements.dualTurnLimit.value = config.dualTurnLimit > 0 ? String(config.dualTurnLimit) : '';
    saveConfig();
  });

  addListener(elements.dualTurnDelay, 'change', (event) => {
    const parsed = Number.parseInt(event.target.value, 10);
    config.dualTurnDelaySeconds = clampNumber(
      Number.isNaN(parsed) ? defaultConfig.dualTurnDelaySeconds : parsed,
      0,
      600,
      defaultConfig.dualTurnDelaySeconds
    );
    elements.dualTurnDelay.value = String(config.dualTurnDelaySeconds);
    saveConfig();
    if (config.dualAutoContinue && isDualChatRunning && !modelAInFlight && !modelBInFlight) {
      scheduleNextDualTurn();
    }
  });

  addListener(elements.agentAName, 'input', (event) => {
    config.agentAName = event.target.value.trim();
    saveConfig();
  });

  addListener(elements.agentBName, 'input', (event) => {
    config.agentBName = event.target.value.trim();
    saveConfig();
  });

  addListener(elements.agentBPrompt, 'input', (event) => {
    config.agentBPrompt = event.target.value;
    saveConfig();
  });

  addListener(elements.ttsPresetSelect, 'change', (event) => {
    handleTtsPresetChange(event.target.value);
  });

  addListener(elements.ttsServerUrl, 'input', (event) => {
    const value = event.target.value.trim();
    config.ttsServerUrl = value;
    userEditedTtsServer = value.length > 0;
    saveConfig();
  });

  addListener(elements.ttsVoiceId, 'input', (event) => {
    const value = event.target.value.trim();
    config.ttsVoiceId = value;
    userEditedTtsVoice = value.length > 0;
    saveConfig();
  });

  addListener(elements.debugToggle, 'change', (event) => {
    config.debugEnabled = event.target.checked;
    saveConfig();
    applyDebugSetting();
  });

  addListener(elements.openDebugButton, 'click', () => {
    openDebugPanel();
  });

  addListener(elements.debugCloseButton, 'click', () => {
    closeDebugPanel();
  });

  addListener(elements.logShutdownButton, 'click', () => {
    const note = prompt('Add optional notes for the shutdown log:')?.trim();
    const message = note?.length ? note : 'Shutdown logged by user.';
    void recordLog('shutdown', message, { silent: false });
  });

  addListener(elements.logErrorButton, 'click', () => {
    const note = prompt('Describe the error to capture in the log:')?.trim();
    const message = note?.length ? note : 'Error logged without description.';
    void recordLog('error', message, { level: 'error', silent: false });
  });

  addListener(elements.exportLogsButton, 'click', () => {
    exportLogs();
  });

  addListener(elements.ttsApiKeyInput, 'input', (event) => {
    config.ttsApiKey = event.target.value;
    saveConfig();
  });

  addListener(elements.ttsVolume, 'input', (event) => {
    config.ttsVolume = clampNumber(event.target.value, 0, 200, config.ttsVolume ?? defaultConfig.ttsVolume);
    elements.ttsVolume.value = String(config.ttsVolume);
    updateTtsVolumeLabel();
    saveConfig();
  });

  addListener(elements.testTtsButton, 'click', () => {
    void speakText(TEST_TTS_PHRASE, { force: true });
  });

  addListener(elements.backgroundUrlInput, 'change', (event) => {
    handleBackgroundUrlChange(event.target.value);
  });

  addListener(elements.backgroundFileInput, 'change', (event) => {
    const [file] = event.target.files || [];
    handleBackgroundFileSelection(file ?? null);
    if (event.target) {
      event.target.value = '';
    }
  });

  addListener(elements.clearBackgroundButton, 'click', () => {
    clearBackgroundImage();
  });

  addListener(elements.chunkSizeInput, 'change', (event) => {
    chunkerState.chunkSize = clampNumber(event.target.value, 100, 2000, 500);
    event.target.value = String(chunkerState.chunkSize);
    const adaptiveOverlap = Math.max(0, Math.round(chunkerState.chunkSize * 0.15));
    chunkerState.chunkOverlap = adaptiveOverlap;
    if (elements.chunkOverlapInput) {
      elements.chunkOverlapInput.value = String(adaptiveOverlap);
    }
  });

  addListener(elements.chunkOverlapInput, 'change', (event) => {
    chunkerState.chunkOverlap = clampNumber(event.target.value, 0, 200, 50);
    event.target.value = String(chunkerState.chunkOverlap);
  });

  addListener(elements.scanChunkerButton, 'click', () => {
    void scanForChunkableFiles();
  });

  addListener(elements.chunkFilesButton, 'click', () => {
    void chunkAllFiles();
  });

  addListener(elements.clearChunksButton, 'click', () => {
    clearAllChunks();
  });

  addListener(window, 'pagehide', () => {
    flushCheckpointBuffer(MODE_CHAT);
    flushCheckpointBuffer(MODE_ARENA);
  });

  // Store listeners for potential cleanup (though not used in this app)
  window._eventListeners = eventListeners;
}

function clampNumber(value, min, max, fallback) {
  const numeric = Number.parseFloat(value);
  if (Number.isNaN(numeric)) {
    console.warn(`clampNumber received invalid value: ${value}, using fallback: ${fallback}`);
    return fallback;
  }
  return Math.min(Math.max(numeric, min), max);
}

function getProviderContextLimit(presetId = config.providerPreset) {
  const targetId = !presetId || presetId === 'inherit' ? config.providerPreset : presetId;
  const preset = providerPresetMap.get(targetId) ?? providerPresetMap.get('custom');
  const candidate = Number(preset?.contextLimit);
  if (Number.isFinite(candidate) && candidate > 0) {
    const normalized = Math.floor(candidate);
    return Math.min(MAX_CONTEXT_LIMIT, Math.max(16, normalized));
  }
  return DEFAULT_CONTEXT_LIMIT;
}

function updateMaxTokensCeiling() {
  if (!elements.maxTokensInput) return;
  const limit = getProviderContextLimit(config.providerPreset);
  elements.maxTokensInput.max = String(limit);
  const current = config.agentAMaxResponseTokens ?? config.maxResponseTokens ?? defaultConfig.agentAMaxResponseTokens;
  const normalized = Math.round(clampNumber(current, 16, limit, current));
  config.agentAMaxResponseTokens = normalized;
  config.maxResponseTokens = normalized; // CODEx: Keep legacy max token mirror updated.
  elements.maxTokensInput.value = String(normalized);
  if (elements.maxTokensHint) {
    elements.maxTokensHint.textContent = `Provider limit: up to ${limit.toLocaleString()} tokens.`;
  }
}

function updateSpeakToggle() {
  elements.speakToggle.checked = Boolean(config.autoSpeak);
}

function updateModelStatus(text, modifier = '', agent = 'A') {
  const target = agent === 'B' ? elements.modelBStatus : elements.modelStatus;
  if (!target) return;
  target.textContent = text;
  target.className = ['status-pill', modifier].filter(Boolean).join(' ');
}

function updateModelConnectionStatus() {
  const descriptorA = buildModelStatusDescriptor('A');
  updateModelStatus(descriptorA.text, descriptorA.modifier, 'A');
  const descriptorB = buildModelStatusDescriptor('B');
  updateModelStatus(descriptorB.text, descriptorB.modifier, 'B');
  updateSystemResourceStatus();
}

function buildModelStatusDescriptor(agent) {
  if (agent === 'B' && !config.agentBEnabled) {
    return { text: 'Model B: Disabled', modifier: 'status-pill--alert' };
  }
  const connection = getAgentConnection(agent);
  const providerId = connection.providerPreset || config.providerPreset || 'custom';
  if (!connection.endpoint || !connection.model) {
    const scope = agent === 'B' && !connection.inherits ? 'arena connection' : 'main connection';
    return {
      text: `Model ${agent}: Not connected (${scope})`,
      modifier: 'status-pill--alert'
    };
  }

  const extras = [];
  if (agent === 'B' && connection.inherits) {
    extras.push('sharing Model A');
  } else if (connection.providerLabel && !connection.providerLabel.toLowerCase().includes('custom')) {
    extras.push(connection.providerLabel);
  }
  const location = describeConnectionLocation(connection.endpoint, providerId);
  if (location) {
    extras.push(location);
  }
  if (isDualChatRunning && (agent === 'A' || agent === 'B')) {
    extras.push('debating now');
  }
  const suffix = extras.length ? ` • ${extras.join(' • ')}` : '';
  return {
    text: `Model ${agent} ready: ${connection.model}${suffix}`,
    modifier: 'status-pill--ready'
  };
}

function describeConnectionLocation(endpoint, providerPresetId = 'custom') {
  if (!endpoint) return '';
  try {
    const url = new URL(endpoint);
    const host = url.hostname.toLowerCase();
    const localHosts = new Set(['localhost', '127.0.0.1', '0.0.0.0']);
    if (localHosts.has(host) || host.endsWith('.local')) {
      return 'Local (RAM/VRAM)';
    }
    if (/^(192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.)/.test(host)) {
      return 'Local network';
    }
    if (providerPresetId && ['lmstudio', 'ollama'].includes(providerPresetId)) {
      return 'Local (RAM/VRAM)';
    }
    if (
      providerPresetId &&
      ['openrouter', 'openai', 'groq', 'together', 'mistral', 'perplexity', 'fireworks', 'deepseek'].includes(providerPresetId)
    ) {
      return 'Cloud hub';
    }
    if (url.protocol === 'https:') {
      return 'Cloud hub';
    }
    return 'Remote';
  } catch (error) {
    return '';
  }
}

function getSystemMemoryEstimate() {
  if (typeof deviceMemoryEstimate === 'undefined') {
    if (typeof navigator !== 'undefined' && 'deviceMemory' in navigator) {
      deviceMemoryEstimate = navigator.deviceMemory;
    } else {
      deviceMemoryEstimate = null;
    }
  }
  if (!deviceMemoryEstimate) return '';
  const rounded = Math.round(deviceMemoryEstimate);
  return rounded ? `${rounded} GB approx.` : '';
}

function ensureGpuRendererInfo() {
  if (typeof gpuRendererInfo !== 'undefined') {
    return gpuRendererInfo;
  }
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (!gl) {
      gpuRendererInfo = null;
      return gpuRendererInfo;
    }
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    if (debugInfo) {
      gpuRendererInfo = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
    } else {
      gpuRendererInfo = gl.getParameter(gl.RENDERER);
    }
  } catch (error) {
    console.warn('Unable to detect GPU information', error);
    gpuRendererInfo = null;
  }
  return gpuRendererInfo;
}

function getGpuDescriptor() {
  const renderer = ensureGpuRendererInfo();
  if (!renderer) return '';
  const cleaned = String(renderer).replace(/\s+/g, ' ').trim();
  if (!cleaned) return '';
  return `${cleaned} (VRAM not exposed)`;
}

function updateSystemResourceStatus() {
  if (!elements.systemResourceStatus) return;
  const parts = [];
  const memoryEstimate = getSystemMemoryEstimate();
  if (memoryEstimate) {
    parts.push(`System RAM: ${memoryEstimate}`);
  }
  const gpu = getGpuDescriptor();
  if (gpu) {
    parts.push(`GPU: ${gpu}`);
  }
  const connectionA = getAgentConnection('A');
  const locationA = describeConnectionLocation(connectionA.endpoint, connectionA.providerPreset || config.providerPreset);
  const modelLocationParts = [];
  if (locationA) {
    modelLocationParts.push(`Model A: ${locationA}`);
  }
  if (config.agentBEnabled) {
    const connectionB = getAgentConnection('B');
    const providerId = connectionB.providerPreset || config.providerPreset;
    const locationB = describeConnectionLocation(connectionB.endpoint, providerId);
    if (locationB) {
      modelLocationParts.push(`Model B: ${locationB}${connectionB.inherits ? ' (shared)' : ''}`);
    }
  } else {
    modelLocationParts.push('Model B: disabled');
  }
  if (modelLocationParts.length) {
    parts.push(modelLocationParts.join(' • '));
  }
  elements.systemResourceStatus.textContent = parts.join(' • ') ||
    'System diagnostics unavailable in this browser.';
}

function populateTtsControls() {
  if (!elements.ttsPresetSelect) return;
  elements.ttsPresetSelect.innerHTML = '';
  for (const preset of ttsPresets) {
    const option = document.createElement('option');
    option.value = preset.id;
    option.textContent = preset.label;
    elements.ttsPresetSelect.append(option);
  }
  if (!ttsPresetMap.has(config.ttsPreset)) {
    config.ttsPreset = defaultConfig.ttsPreset;
  }
  elements.ttsPresetSelect.value = config.ttsPreset;
  renderTtsPresetList();
  populateBrowserVoices();
  if ('speechSynthesis' in window) {
    window.speechSynthesis.onvoiceschanged = populateBrowserVoices;
  }
}

function renderTtsPresetList() {
  if (!elements.ttsPresetList) return;
  elements.ttsPresetList.innerHTML = '';
  for (const preset of ttsPresets) {
    const item = document.createElement('li');
    const title = document.createElement('strong');
    title.textContent = preset.label;
    item.append(title);

    const metaParts = [preset.rating, preset.hardware, preset.languages].filter(Boolean);
    if (metaParts.length) {
      const meta = document.createElement('span');
      meta.className = 'preset-meta';
      meta.textContent = metaParts.join(' • ');
      item.append(meta);
    }

    if (preset.description || preset.notes) {
      const description = document.createElement('p');
      description.className = 'preset-description';
      description.textContent = [preset.description, preset.notes].filter(Boolean).join(' ');
      item.append(description);
    }

    elements.ttsPresetList.append(item);
  }
}

function populateBrowserVoices() {
  if (!elements.ttsVoiceCatalog || !('speechSynthesis' in window)) return;
  const voices = window.speechSynthesis.getVoices();
  if (!voices.length) return;
  elements.ttsVoiceCatalog.innerHTML = '';
  for (const voice of voices) {
    const option = document.createElement('option');
    option.value = voice.name;
    option.textContent = `${voice.name} (${voice.lang})`;
    elements.ttsVoiceCatalog.append(option);
  }
}

function handleTtsPresetChange(presetId) {
  const preset = getTtsPreset(presetId) ?? getTtsPreset(defaultConfig.ttsPreset);
  config.ttsPreset = preset.id;
  if (elements.ttsPresetSelect) {
    elements.ttsPresetSelect.value = config.ttsPreset;
  }
  if (preset.defaultUrl && (!userEditedTtsServer || !config.ttsServerUrl)) {
    config.ttsServerUrl = preset.defaultUrl;
    if (elements.ttsServerUrl) {
      elements.ttsServerUrl.value = config.ttsServerUrl;
    }
  }
  if (preset.voiceId && (!userEditedTtsVoice || !config.ttsVoiceId)) {
    config.ttsVoiceId = preset.voiceId;
    if (elements.ttsVoiceId) {
      elements.ttsVoiceId.value = config.ttsVoiceId;
    }
  }
  updateTtsPresetDetails();
  saveConfig();
}

function updateTtsPresetDetails() {
  if (!elements.ttsPresetDetails) return;
  const preset = getTtsPreset();
  if (!preset) {
    elements.ttsPresetDetails.textContent = 'Select a voice preset to see setup tips.';
    setElementVisibility(elements.ttsServerField, false);
    setElementVisibility(elements.ttsApiKeyField, false);
    return;
  }

  const metaParts = [preset.rating, preset.hardware, preset.languages].filter(Boolean);
  const detailText = [preset.description, preset.notes, metaParts.length ? `(${metaParts.join(' • ')})` : '']
    .filter(Boolean)
    .join(' ');
  elements.ttsPresetDetails.textContent = detailText;

  if (elements.ttsVoiceId) {
    elements.ttsVoiceId.placeholder = preset.voiceHint ?? 'Voice or speaker ID';
  }
  if (elements.ttsServerUrl) {
    elements.ttsServerUrl.placeholder = preset.serverHint ?? preset.defaultUrl ?? 'http://localhost:5002';
  }

  setElementVisibility(elements.ttsServerField, preset.provider !== 'browser');
  setElementVisibility(elements.ttsApiKeyField, Boolean(preset.needsApiKey));
  setElementVisibility(elements.ttsVoiceField, true);
}

function updateTtsVolumeLabel() {
  if (!elements.ttsVolumeValue) return;
  const percent = clampNumber(config.ttsVolume ?? defaultConfig.ttsVolume, 0, 200, defaultConfig.ttsVolume);
  elements.ttsVolumeValue.textContent = `${percent}%`;
}

function setElementVisibility(node, shouldShow) {
  if (!node) return;
  node.hidden = !shouldShow;
}

function getTtsPreset(presetId = config.ttsPreset) {
  return ttsPresetMap.get(presetId);
}

async function initDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('chatDatabase', 1);

    request.onerror = (event) => {
      console.error('IndexedDB error', event);
      reject(event);
    };

    request.onupgradeneeded = (event) => {
      const database = event.target.result;
      if (!database.objectStoreNames.contains('messages')) {
        database.createObjectStore('messages', { keyPath: 'timestamp' });
      }
      if (!database.objectStoreNames.contains(RAG_STORE_NAME)) {
        database.createObjectStore(RAG_STORE_NAME, { keyPath: 'id' });
      }
      if (!database.objectStoreNames.contains(LOG_STORE_NAME)) {
        database.createObjectStore(LOG_STORE_NAME, { keyPath: 'id' });
      }
      if (!database.objectStoreNames.contains(REFLEX_HISTORY_STORE)) {
        database.createObjectStore(REFLEX_HISTORY_STORE, { keyPath: 'id', autoIncrement: true });
      }
    };

    request.onsuccess = (event) => {
      db = event.target.result;
      resolve();
    };
  });
}

async function loadConversationFromStorage() {
  try {
    const storedMessages = await getAllMessages();
    storedMessages.sort((a, b) => a.timestamp - b.timestamp);
    const chatEntries = storedMessages.filter((entry) => (entry.mode ?? MODE_CHAT) === MODE_CHAT);
    const arenaEntries = storedMessages.filter((entry) => (entry.mode ?? MODE_CHAT) === MODE_ARENA);

    for (const entry of chatEntries.slice(-150)) {
      const normalizedTimestamp = normalizeTimestamp(entry.timestamp);
      const record = {
        id: entry.id ?? normalizedTimestamp,
        role: entry.role,
        content: entry.content,
        timestamp: normalizedTimestamp,
        origin: entry.origin ?? 'long-term',
        mode: MODE_CHAT
      };
      assignTurnNumber(record, entry.turnNumber);
      const normalizedId = normalizeMessageId(record.id);
      const isPinned = pinnedMessageIds.has(normalizedId) || Boolean(entry.pinned);
      if (isPinned) {
        record.pinned = true;
        pinnedMessageIds.add(normalizedId);
      }
      conversationLog.push(record);
      renderMessage(record, { badge: 'long-term' });
      floatingMemory.push(record);
    }

    for (const entry of arenaEntries.slice(-MAX_ARENA_RAG_MESSAGES)) {
      const normalizedTimestamp = normalizeTimestamp(entry.timestamp);
      const id = entry.id ?? `arena-${normalizedTimestamp}`;
      const normalizedId = normalizeMessageId(id);
      const isPinned = pinnedMessageIds.has(normalizedId) || Boolean(entry.pinned);
      const memoryRecord = {
        id,
        role: entry.role ?? 'arena',
        content: entry.content,
        timestamp: normalizedTimestamp,
        origin: entry.origin ?? 'arena',
        turnNumber: entry.turnNumber ?? null,
        pinned: isPinned,
        mode: MODE_ARENA
      };
      if (isPinned) {
        pinnedMessageIds.add(normalizedId);
      }
      floatingMemory.push(memoryRecord);
    }
    trimFloatingMemory();
    renderFloatingMemoryWorkbench();
    persistPinnedMessages();
    await persistChatRagSnapshot();
  } catch (error) {
    console.error('Failed to load history', error);
  }
}

function getAllMessages() {
  if (!db) return Promise.resolve([]);
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['messages'], 'readonly');
    const store = transaction.objectStore('messages');
    const request = store.getAll();
    request.onerror = (event) => reject(event);
    request.onsuccess = () => resolve(request.result || []);
  });
}

function getAllRagRecords() {
  if (!db) return Promise.resolve([]);
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([RAG_STORE_NAME], 'readonly');
    const store = transaction.objectStore(RAG_STORE_NAME);
    const request = store.getAll();
    request.onerror = (event) => reject(event);
    request.onsuccess = () => resolve(request.result || []);
  });
}

function getAllLogs() {
  if (!db) return Promise.resolve([]);
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([LOG_STORE_NAME], 'readonly');
    const store = transaction.objectStore(LOG_STORE_NAME);
    const request = store.getAll();
    request.onerror = (event) => reject(event);
    request.onsuccess = () => resolve(request.result || []);
  });
}

async function loadLogsFromStorage() {
  if (!db) return;
  try {
    const stored = await getAllLogs();
    const sorted = stored
      .filter((entry) => entry && entry.id && entry.timestamp)
      .sort((a, b) => a.timestamp - b.timestamp);
    logEntries = sorted.slice(-250);
    updateLogList();
  } catch (error) {
    console.error('Failed to load logs', error);
    logEntries = [];
    updateLogList();
  }
}

async function persistLogEntry(entry) {
  if (!db) return;
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([LOG_STORE_NAME], 'readwrite');
    const store = transaction.objectStore(LOG_STORE_NAME);
    const payload = { ...entry, timestamp: normalizeTimestamp(entry.timestamp) };
    const request = store.put(payload);
    request.onsuccess = () => resolve();
    request.onerror = (event) => {
      console.error('Failed to persist log entry', event);
      reject(event);
    };
  });
}

async function recordLog(eventType, message, options = {}) {
  const { level = 'info', silent = false } = options;
  const entry = {
    id: `${eventType}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    eventType,
    message,
    level,
    timestamp: Date.now()
  };
  logEntries = [...logEntries, entry].sort((a, b) => a.timestamp - b.timestamp).slice(-250);
  updateLogList();
  try {
    await persistLogEntry(entry);
    if (!silent) {
      logStatusMessage(`Logged ${eventType}.`);
    }
  } catch (error) {
    if (!silent) {
      logStatusMessage('Failed to persist log entry.');
    }
  }
  return entry;
}

function exportLogs() {
  if (!logEntries.length) {
    logStatusMessage('No logs available to export yet.');
    return;
  }
  const ordered = [...logEntries].sort((a, b) => a.timestamp - b.timestamp);
  const content = ordered
    .map((entry) => {
      const iso = new Date(entry.timestamp).toISOString();
      return `${iso}\t${entry.eventType}\t${entry.level}\t${entry.message}`;
    })
    .join('\n');
  const blob = new Blob([content], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `logs/${LOG_EXPORT_PREFIX}-${new Date().toISOString().replace(/[:.]/g, '-')}.txt`;
  anchor.click();
  URL.revokeObjectURL(url);
  logStatusMessage('Logs exported.');
}

function updateLogList() {
  if (!elements.logList) return;
  elements.logList.innerHTML = '';
  if (!logEntries.length) {
    const empty = document.createElement('li');
    empty.textContent = 'No log entries captured yet.';
    elements.logList.appendChild(empty);
    return;
  }
  const recent = [...logEntries].sort((a, b) => b.timestamp - a.timestamp).slice(0, 30);
  for (const entry of recent) {
    elements.logList.appendChild(createLogListItem(entry));
  }
}

function createLogListItem(entry) {
  const item = document.createElement('li');
  const isError = entry.level === 'error' || entry.eventType === 'error';
  const isShutdown = entry.eventType === 'shutdown';
  if (isError) {
    item.classList.add('debug-log--error');
  } else if (isShutdown) {
    item.classList.add('debug-log--shutdown');
  }
  const timestamp = document.createElement('time');
  const normalized = normalizeTimestamp(entry.timestamp);
  timestamp.dateTime = new Date(normalized).toISOString();
  timestamp.textContent = new Date(normalized).toLocaleString();
  const message = document.createElement('p');
  message.textContent = `[${entry.eventType.toUpperCase()}] ${entry.message}`;
  item.append(timestamp, message);
  return item;
}

function logStatusMessage(text) {
  if (!elements.logStatus) return;
  elements.logStatus.textContent = text;
  if (logStatusTimer) {
    clearTimeout(logStatusTimer);
  }
  if (!text) return;
  logStatusTimer = setTimeout(() => {
    if (elements.logStatus.textContent === text) {
      elements.logStatus.textContent = '';
    }
  }, LOG_STATUS_TIMEOUT);
}

function initProcessRegistry() {
  processRegistry = new Map(PROCESS_BASELINE.map((item) => [item.id, { ...item }]));
  updateProcessList();
}

function applyDebugSetting() {
  document.body.classList.toggle('debug-enabled', Boolean(config.debugEnabled));
  if (elements.debugToggle) {
    elements.debugToggle.checked = Boolean(config.debugEnabled);
  }
  if (!config.debugEnabled) {
    closeDebugPanel();
  }
  updateProcessList();
}

function updateProcessState(id, updates = {}) {
  const existing = processRegistry.get(id) ?? { id, label: id, status: 'Idle', detail: '' };
  const next = { ...existing, ...updates };
  processRegistry.set(id, next);
  updateProcessList();
}

function updateProcessList() {
  if (!elements.processList) return;
  elements.processList.innerHTML = '';
  const values = [...processRegistry.values()];
  if (!values.length) {
    const empty = document.createElement('li');
    empty.textContent = config.debugEnabled
      ? 'No tracked processes yet.'
      : 'Enable the debug console to track live processes.';
    elements.processList.appendChild(empty);
    return;
  }
  for (const item of values) {
    const node = document.createElement('li');
    const heading = document.createElement('strong');
    heading.textContent = item.label;
    const status = document.createElement('span');
    status.className = 'debug-process-status';
    status.textContent = item.status;
    heading.appendChild(status);
    const detail = document.createElement('p');
    detail.className = 'debug-process-detail';
    detail.textContent = item.detail || '';
    node.append(heading, detail);
    elements.processList.appendChild(node);
  }
}

function openDebugPanel() {
  if (!config.debugEnabled) {
    logStatusMessage('Enable the debug console to open diagnostics.');
    return;
  }
  if (!elements.debugPanel) return;
  debugPanelVisible = true;
  elements.debugPanel.hidden = false;
  updateProcessList();
  updateLogList();
}

function closeDebugPanel() {
  if (!elements.debugPanel) return;
  elements.debugPanel.hidden = true;
  debugPanelVisible = false;
}

async function fetchStaticRagRecords() {
  const records = [];
  let filesLoaded = 0;
  let bytesLoaded = 0;
  const errors = [];

  for (const descriptor of STATIC_RAG_MANIFESTS) {
    const manifest = await fetchRagManifest(descriptor).catch((error) => {
      console.warn('Failed to load RAG manifest', descriptor.url, error);
      errors.push({ descriptor, message: error.message });
      return null;
    });
    if (!manifest) continue;
    const entries = normalizeRagManifestEntries(manifest);
    for (const entry of entries) {
      try {
        const record = await loadRagManifestEntry(descriptor, entry);
        if (!record || !Array.isArray(record.messages) || !record.messages.length) continue;
        records.push(record);
        filesLoaded += 1;
        bytesLoaded += record.bytes ?? estimateMessagesSize(record.messages);
      } catch (error) {
        console.error('Failed to ingest static RAG file', entry, error);
        errors.push({ descriptor, entry, message: error.message });
      }
    }
  }

  const archiveEntries = listRamDiskArchiveEntries();
  for (const { path, data } of archiveEntries) {
    if (!path || !data) continue;
    const entry = data.entry || {};
    const content = typeof data.content === 'string' ? data.content : '';
    const mode = entry.mode || MODE_CHAT;
    const origin = entry.origin || 'rag-archive';
    const role = entry.role || 'archive';
    const label = entry.label || path.replace(/^rag\//, '');
    const recordSlug = slugify(entry.id || path) || `archive-${Date.now()}`;
    const recordId = entry.id || `ram-archive-${recordSlug}`;
    const timestamp = normalizeTimestamp(entry.timestamp ?? data.savedAt ?? Date.now());
    const message = {
      id: `${recordId}-entry`,
      role,
      content,
      timestamp,
      origin,
      mode,
      pinned: Boolean(entry.pinned)
    };
    const bytes = content ? new Blob([content]).size : estimateMessagesSize([message]);
    records.push({ id: recordId, label, mode, origin, messages: [message], bytes });
    filesLoaded += 1;
    bytesLoaded += bytes;
  }

  return { records, filesLoaded, bytesLoaded, errors };
}

async function fetchRagManifest(descriptor) {
  if (!descriptor?.url) return null;
  try {
    const response = await fetch(descriptor.url, { cache: 'no-store' });
    if (response.ok) {
      const manifest = await response.json();
      cacheManifestInRam(descriptor, manifest);
      persistManifestToStorage(descriptor, manifest);
      return manifest;
    }
    if (response.status !== 404) {
      console.warn(`RAG manifest ${descriptor.url} returned ${response.status}`);
    }
  } catch (error) {
    console.warn('Unable to fetch RAG manifest', descriptor.url, error);
  }
  return readManifestFromRam(descriptor) || readManifestFromStorage(descriptor);
}

function normalizeRagManifestEntries(manifest) {
  if (!manifest) return [];
  if (Array.isArray(manifest)) return manifest;
  if (Array.isArray(manifest.entries)) return manifest.entries;
  if (Array.isArray(manifest.files)) {
    return manifest.files.map((item) => (typeof item === 'string' ? { path: item } : item));
  }
  if (manifest.entries && typeof manifest.entries === 'object') {
    return Object.values(manifest.entries);
  }
  return [];
}

async function loadRagManifestEntry(descriptor, entry) {
  if (!entry || entry.disabled) return null;
  const basePath = descriptor.basePath || descriptor.url.replace(/[^/]+$/, '');
  const mode = entry.mode || descriptor.defaultMode || MODE_CHAT;
  const origin = entry.origin || (mode === MODE_ARENA ? 'rag-arena' : 'rag-file');
  const role = entry.role || entry.speaker || 'archive';
  const label = entry.label || entry.title || descriptor.label || 'RAG import';
  const baseIdSource = entry.id || entry.path || entry.file || entry.source || entry.label || entry.title || label;
  const baseSlug = slugify(baseIdSource) || `external-${Date.now()}`;
  const recordId = entry.id || `rag-${baseSlug}`;
  const timestamp = normalizeTimestamp(entry.timestamp ?? Date.now());
  const pinned = Boolean(entry.pinned);

  if (Array.isArray(entry.messages) && entry.messages.length) {
    const messages = entry.messages
      .map((item, index) =>
        coerceExternalMessage(item, {
          baseId: `${recordId}-inline`,
          role,
          origin,
          mode,
          timestamp: timestamp + index,
          pinned
        })
      )
      .filter(Boolean);
    return { id: recordId, label, mode, origin, messages, bytes: estimateMessagesSize(messages) };
  }

  const path = entry.path || entry.file || entry.source;
  if (!path) {
    return null;
  }

  const resolvedPath = resolveRagPath(basePath, path);
  const format = ((entry.format || '').toLowerCase() || path.split('.').pop() || '').toLowerCase();

  if (SUPPORTED_RAG_JSON_FORMATS.has(format)) {
    const response = await fetch(resolvedPath, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} while reading ${path}`);
    }
    const text = await response.text();
    const messages = normalizeExternalMessagesFromJson(text, {
      baseId: recordId,
      role,
      origin,
      mode,
      pinned
    });
    return { id: recordId, label, mode, origin, messages, bytes: new Blob([text]).size };
  }

  if (SUPPORTED_RAG_TEXT_FORMATS.has(format) || !format) {
    const response = await fetch(resolvedPath, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} while reading ${path}`);
    }
    const text = await response.text();
    const content = text.trim();
    if (!content) return null;
    const message = {
      id: `${recordId}-full`,
      role,
      content,
      timestamp,
      origin,
      mode,
      pinned
    };
    return { id: recordId, label, mode, origin, messages: [message], bytes: new Blob([text]).size };
  }

  if (SUPPORTED_RAG_BINARY_FORMATS.has(format)) {
    const response = await fetch(resolvedPath, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} while reading ${path}`);
    }
    const buffer = await response.arrayBuffer();
    const text = await extractPdfText(buffer).catch((error) => {
      throw new Error(`PDF parse failed: ${error.message}`);
    });
    const content = text.trim();
    if (!content) return null;
    const message = {
      id: `${recordId}-pdf`,
      role,
      content,
      timestamp,
      origin,
      mode,
      pinned
    };
    return { id: recordId, label, mode, origin, messages: [message], bytes: buffer.byteLength };
  }

  const response = await fetch(resolvedPath, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} while reading ${path}`);
  }
  const fallbackText = await response.text();
  const fallbackContent = fallbackText.trim();
  if (!fallbackContent) return null;
  const message = {
    id: `${recordId}-raw`,
    role,
    content: fallbackContent,
    timestamp,
    origin,
    mode,
    pinned
  };
  return { id: recordId, label, mode, origin, messages: [message], bytes: new Blob([fallbackText]).size };
}

function resolveRagPath(basePath, filePath) {
  if (!filePath) return basePath;
  if (/^https?:/i.test(filePath)) return filePath;
  if (filePath.startsWith('/')) {
    return filePath.replace(/^\//, '');
  }
  if (filePath.startsWith('rag/')) {
    return filePath;
  }
  const normalizedBase = basePath.endsWith('/') ? basePath : `${basePath}/`;
  const combinedPath = `${normalizedBase}${filePath}`;
  return combinedPath.split('\\').join('/');
}

function normalizeExternalMessagesFromJson(text, defaults = {}) {
  const messages = [];
  if (!text) return messages;
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    parsed = null;
  }

  if (!parsed) {
    const lines = text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith('//'));
    const jsonl = [];
    for (const line of lines) {
      try {
        jsonl.push(JSON.parse(line));
      } catch (error) {
        console.warn('Skipping malformed JSONL line in RAG file:', error.message);
      }
    }
    parsed = jsonl;
  }

  const groups = [];
  if (Array.isArray(parsed)) {
    groups.push(parsed);
  } else if (parsed && typeof parsed === 'object') {
    if (Array.isArray(parsed.messages)) {
      groups.push(parsed.messages);
    }
    if (Array.isArray(parsed.turns)) {
      groups.push(parsed.turns);
    }
    if (!groups.length) {
      groups.push([parsed]);
    }
  }

  let sequence = 0;
  for (const group of groups) {
    for (const raw of group) {
      const message = coerceExternalMessage(raw, {
        ...defaults,
        baseId: `${defaults.baseId || 'rag'}-json`,
        sequence
      });
      if (message) {
        messages.push(message);
        sequence += 1;
      }
    }
  }
  return messages;
}

function estimateMessagesSize(messages = []) {
  return messages.reduce((total, item) => total + approximateSize(item), 0);
}

function estimateTokenCount(text) {
  if (!text) return 0;
  const parts = text
    .trim()
    .split(/\s+/)
    .filter(Boolean);
  let total = 0;
  for (const part of parts) {
    total += Math.max(1, Math.ceil(part.length / 4));
  }
  return total;
}

async function ensurePdfModule() {
  if (pdfLoaderPromise) return pdfLoaderPromise;
  pdfLoaderPromise = import(`${PDFJS_CDN_BASE}/legacy/build/pdf.mjs`)
    .then((module) => {
      if (module.GlobalWorkerOptions) {
        module.GlobalWorkerOptions.workerSrc = `${PDFJS_CDN_BASE}/legacy/build/pdf.worker.min.js`;
      }
      return module;
    })
    .catch((error) => {
      console.warn('Failed to load pdf.js helper', error);
      return null;
    });
  return pdfLoaderPromise;
}

async function extractPdfText(buffer) {
  const pdfjs = await ensurePdfModule();
  if (!pdfjs) {
    throw new Error('PDF reader unavailable');
  }
  const documentInstance = await pdfjs.getDocument({ data: buffer }).promise;
  let text = '';
  for (let pageNumber = 1; pageNumber <= documentInstance.numPages; pageNumber += 1) {
    const page = await documentInstance.getPage(pageNumber);
    const content = await page.getTextContent();
    const strings = content.items?.map((item) => item.str).filter(Boolean) ?? [];
    text += strings.join(' ') + '\n';
  }
  return text;
}

function coerceExternalMessage(raw, defaults = {}) {
  if (raw == null) return null;
  if (typeof raw === 'string') {
    const content = raw.trim();
    if (!content) return null;
    return {
      id: `${defaults.baseId || 'rag'}-${defaults.sequence ?? Date.now()}`,
      role: defaults.role || 'archive',
      content,
      timestamp: normalizeTimestamp(defaults.timestamp ?? Date.now()),
      origin: defaults.origin || 'rag',
      turnNumber: defaults.turnNumber ?? null,
      pinned: Boolean(defaults.pinned),
      mode: defaults.mode || MODE_CHAT
    };
  }

  if (typeof raw !== 'object') {
    return null;
  }

  let contentValue = raw.content ?? raw.text ?? raw.body ?? '';
  if (Array.isArray(contentValue)) {
    contentValue = contentValue
      .map((item) => {
        if (typeof item === 'string') return item;
        if (item && typeof item === 'object') {
          if (typeof item.text === 'string') return item.text;
          if (typeof item.value === 'string') return item.value;
        }
        return '';
      })
      .filter(Boolean)
      .join('\n');
  }
  const content = contentValue.toString();
  const trimmed = content.trim();
  if (!trimmed) {
    return null;
  }

  const message = {
    id:
      raw.id ||
      raw.uuid ||
      raw.key ||
      `${defaults.baseId || 'rag'}-${defaults.sequence ?? Date.now()}`,
    role: raw.role || raw.speaker || defaults.role || 'archive',
    content: trimmed,
    timestamp: normalizeTimestamp(raw.timestamp ?? raw.time ?? defaults.timestamp ?? Date.now()),
    origin: raw.origin || defaults.origin || 'rag',
    turnNumber: raw.turnNumber ?? raw.turn ?? defaults.turnNumber ?? null,
    pinned: Boolean(raw.pinned ?? defaults.pinned),
    mode: raw.mode || defaults.mode || MODE_CHAT
  };

  if (raw.speakerName && !message.speakerName) {
    message.speakerName = raw.speakerName;
  }
  if (raw.metadata && !message.metadata) {
    message.metadata = raw.metadata;
  }
  if (Array.isArray(raw.embedding)) {
    message.embedding = raw.embedding;
  }
  return message;
}

function slugify(value) {
  if (value == null) return '';
  return value
    .toString()
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function buildFloatingSeenSet() {
  const seen = new Set();
  for (const item of floatingMemory) {
    const id = normalizeMessageId(item.id ?? item.timestamp);
    if (id) {
      seen.add(id);
    }
  }
  return seen;
}

function ingestRagRecords(records, seen = buildFloatingSeenSet()) {
  let added = 0;
  if (!Array.isArray(records)) {
    return { added, seen };
  }
  for (const record of records) {
    if (!record || !Array.isArray(record.messages)) continue;
    for (const message of record.messages) {
      const entry = coerceExternalMessage(message, {
        baseId: record.id,
        role: message.role || record.role || (record.mode === MODE_ARENA ? 'arena' : 'memory'),
        origin: message.origin || record.origin || 'rag-indexed',
        mode: message.mode || record.mode || MODE_CHAT,
        pinned: message.pinned,
        timestamp: message.timestamp
      });
      if (!entry || !entry.content) continue;
      const normalizedId = normalizeMessageId(entry.id ?? entry.timestamp);
      if (seen.has(normalizedId)) continue;
      if (entry.pinned) {
        pinnedMessageIds.add(normalizedId);
      }
      if (Array.isArray(entry.embedding)) {
        messageEmbeddingCache.set(normalizedId, entry.embedding);
        delete entry.embedding;
      } else if (Array.isArray(message.embedding)) {
        messageEmbeddingCache.set(normalizedId, message.embedding);
      } else if (message.metadata?.embeddingKey) {
        const cachedVector = resolveCachedEmbedding(EMBEDDING_BUCKET_KEYS.SHARED, message.metadata.embeddingKey); // CODEx: Pull shared chunk embedding when metadata links provided.
        if (Array.isArray(cachedVector)) {
          messageEmbeddingCache.set(normalizedId, cachedVector);
        }
      }
      floatingMemory.push(entry);
      seen.add(normalizedId);
      added += 1;
    }
  }
  if (added > 0) {
    trimFloatingMemory();
    persistPinnedMessages();
  }
  return { added, seen };
}

async function hydrateFloatingMemoryFromRag() {
  try {
    const seen = buildFloatingSeenSet();
    const [indexedRecords, staticBundle, chunkedRecords] = await Promise.all([
      db
        ? getAllRagRecords().catch((error) => {
            console.error('Failed to read IndexedDB RAG records', error);
            void recordLog('error', `Failed to read IndexedDB RAG records: ${error.message}`, { level: 'error' });
            return [];
          })
        : [],
      fetchStaticRagRecords().catch((error) => {
          console.error('Failed to fetch static RAG records', error);
          void recordLog('error', `Failed to fetch static RAG records: ${error.message}`, { level: 'error' });
          return { records: [], filesLoaded: 0, bytesLoaded: 0, errors: [{ message: error.message }] };
        }),
      loadChunkedRagRecords().catch((error) => {
          console.error('Failed to load chunked RAG records', error);
          void recordLog('error', `Failed to load chunked RAG records: ${error.message}`, { level: 'error' });
          return [];
        })
    ]);

    const combinedRecords = [];
    if (Array.isArray(indexedRecords) && indexedRecords.length) {
      for (const record of indexedRecords) {
        if (!record || !Array.isArray(record.messages) || !record.messages.length) continue;
        combinedRecords.push({
          id: record.id || `indexed-${combinedRecords.length + 1}`,
          mode: record.mode ?? MODE_CHAT,
          origin: record.mode === MODE_ARENA ? 'arena-archive' : record.origin ?? 'rag-indexed',
          messages: record.messages
        });
      }
    }

    if (Array.isArray(staticBundle.records) && staticBundle.records.length) {
      combinedRecords.push(...staticBundle.records);
    }

    if (Array.isArray(chunkedRecords) && chunkedRecords.length) {
      combinedRecords.push(...chunkedRecords);
    }

    if (!combinedRecords.length) {
      ragTelemetry.lastLoad = null;
      ragTelemetry.lastLoadCount = 0;
      ragTelemetry.lastLoadRecords = 0;
      ragTelemetry.lastLoadBytes = 0;
      ragTelemetry.staticFiles = 0;
      ragTelemetry.staticBytes = 0;
      ragTelemetry.staticErrors = staticBundle.errors || [];
      ragTelemetry.loaded = false;
      updateRagStatusDisplay();
      return;
    }

    const { added } = ingestRagRecords(combinedRecords, seen);

    ragTelemetry.lastLoad = Date.now();
    ragTelemetry.lastLoadCount = added;
    ragTelemetry.lastLoadRecords = combinedRecords.length;
    ragTelemetry.lastLoadBytes = calculateRagMemoryUsage();
    ragTelemetry.staticFiles = staticBundle.filesLoaded || 0;
    ragTelemetry.staticBytes = staticBundle.bytesLoaded || 0;
    ragTelemetry.staticErrors = staticBundle.errors || [];
    ragTelemetry.loaded = combinedRecords.length > 0;

    renderFloatingMemoryWorkbench();

    if (added > 0) {
      const snippetLabel = added === 1 ? 'memory snippet' : 'memory snippets';
      const sourceLabel = combinedRecords.length === 1 ? 'source' : 'sources';
      const fileNote =
        ragTelemetry.staticFiles > 0
          ? ` (${ragTelemetry.staticFiles} static file${ragTelemetry.staticFiles === 1 ? '' : 's'} scanned)`
          : '';
      addSystemMessage(`Recovered ${added} ${snippetLabel} from ${combinedRecords.length} ${sourceLabel}${fileNote} into the floating buffer.`);
    } else if (!ragTelemetry.loaded) {
      addSystemMessage('Checked RAG archives but no saved memories were available.');
    } else {
      updateFloatingSummary();
    }

    if (ragTelemetry.staticErrors.length) {
      const errorLabel = ragTelemetry.staticErrors.length === 1 ? 'file' : 'files';
      addSystemMessage(`Skipped ${ragTelemetry.staticErrors.length} RAG ${errorLabel}. Open the debug console for details.`);
      void recordLog('error', `Skipped ${ragTelemetry.staticErrors.length} RAG ${errorLabel} during hydration`, { level: 'error' });
    }
  } catch (error) {
    console.error('Failed to hydrate floating memory from RAG', error);
    void recordLog('error', `Failed to hydrate floating memory from RAG: ${error.message}`, { level: 'error' });
    addSystemMessage('Failed to load RAG memories. Check the debug console for details.');
  }
}

async function loadChunkedRagRecords() {
  const records = [];
  try {
    const seen = new Set();

    const [manifest, chunkedListing] = await Promise.all([
      fetchRagManifest(STATIC_RAG_MANIFESTS[0]).catch((error) => {
        console.warn('Could not load RAG manifest for chunked files:', error);
        void recordLog('error', `Could not load RAG manifest for chunked files: ${error.message}`, { level: 'error' });
        return null;
      }),
      listRagDirectoryEntries('rag/chunked/')
    ]);

    const manifestEntries = normalizeRagManifestEntries(manifest);
    const manifestChunkedPaths = manifestEntries
      .map((entry) => (entry && entry.path ? toChunkedPath(resolveRagPath('rag/', entry.path)) : null))
      .filter(Boolean);

    const candidatePaths = filterChunkablePaths(dedupePaths([...manifestChunkedPaths, ...chunkedListing]));

    for (const chunkedPath of candidatePaths) {
      try {
        const result = await loadChunkedFileData(chunkedPath, fromChunkedPath(chunkedPath));
        const chunkList = result?.data?.chunks;
        if (!Array.isArray(chunkList) || chunkList.length === 0) {
          continue;
        }
        const originalPath = result?.data?.originalPath || fromChunkedPath(chunkedPath);
        const canonical = canonicalizeOriginalPath(originalPath || chunkedPath);
        if (seen.has(canonical)) {
          continue;
        }
        const messages = chunkList.map((chunk, index) => ({
          id: `${originalPath}-chunk-${index}`,
          role: 'archive',
          content: chunk.content,
          timestamp: Date.now() + index,
          origin: 'rag-chunked',
          mode: MODE_CHAT,
          pinned: false
        }));
        const labelSuffix = result.source === 'ram' ? ' (RAM disk)' : '';
        records.push({
          id: originalPath,
          label: `Chunked: ${originalPath}${labelSuffix}`,
          mode: MODE_CHAT,
          origin: 'rag-chunked',
          messages,
          bytes: estimateMessagesSize(messages)
        });
        seen.add(canonical);
        const locationLabel = result.source === 'ram' ? 'RAM disk' : 'workspace';
        addSystemMessage(`✓ Loaded chunked file: ${originalPath.replace(/^rag\//, '')} (${chunkList.length} chunks, ${locationLabel})`);
      } catch (fileError) {
        console.warn(`Could not load chunked file ${chunkedPath}:`, fileError);
        void recordLog('error', `Could not load chunked file ${chunkedPath}: ${fileError.message}`, { level: 'error' });
      }
    }

    for (const { path, data } of listRamDiskChunkedEntries()) {
      const originalPath = typeof data?.originalPath === 'string' ? data.originalPath : fromChunkedPath(path);
      const canonical = canonicalizeOriginalPath(originalPath || path);
      if (!originalPath || seen.has(canonical) || !isSupportedChunkablePath(originalPath)) {
        continue;
      }
      const chunkList = Array.isArray(data?.chunks) ? data.chunks : [];
      if (!chunkList.length) {
        continue;
      }
      const messages = chunkList.map((chunk, index) => {
        const key = `${canonical}#${index}`;
        const vector = Array.isArray(chunk.embedding)
          ? chunk.embedding
          : resolveCachedEmbedding(EMBEDDING_BUCKET_KEYS.SHARED, key) || null; // CODEx: Rehydrate shared embeddings when RAM disk omits inline vectors.
        if (Array.isArray(vector)) {
          storeChunkEmbedding(key, vector, EMBEDDING_BUCKET_KEYS.SHARED); // CODEx: Refresh shared cache footprint when hydrating from RAM disk.
        }
        return {
          id: `${originalPath}-chunk-${index}`,
          role: 'archive',
          content: chunk.content,
          timestamp: Date.now() + index,
          origin: 'rag-chunked',
          mode: MODE_CHAT,
          pinned: false,
          embedding: vector,
          metadata: { embeddingKey: key }
        };
      });

      records.push({
        id: originalPath,
        label: `Chunked: ${originalPath} (RAM disk)`,
        mode: MODE_CHAT,
        origin: 'rag-chunked',
        messages,
        bytes: estimateMessagesSize(messages)
      });
      seen.add(canonical);
    }
  } catch (error) {
    console.error('Error loading chunked RAG records:', error);
    void recordLog('error', `Error loading chunked RAG records: ${error.message}`, { level: 'error' });
  }

  return records;
}

async function manualRagReload() {
  const previousLoad = ragTelemetry.lastLoad;
  await hydrateFloatingMemoryFromRag();
  if (ragTelemetry.lastLoadCount === 0) {
    let message = previousLoad
      ? 'RAG archives refreshed. No new memories were added this time.'
      : 'Checked RAG archives. No saved memories were available to load.';
    if (ragTelemetry.staticErrors && ragTelemetry.staticErrors.length) {
      const errorLabel = ragTelemetry.staticErrors.length === 1 ? 'file' : 'files';
      message += ` Skipped ${ragTelemetry.staticErrors.length} ${errorLabel}; open the debug console for more detail.`;
    }
    addSystemMessage(message);
  }
}

async function manualChunkReload() {
  try {
    const seen = buildFloatingSeenSet();
    const chunkedRecords = await loadChunkedRagRecords();
    if (!Array.isArray(chunkedRecords) || chunkedRecords.length === 0) {
      addSystemMessage('No chunked files found to load into floating memory.');
      return;
    }
    const priorRagCount = countRagMemories();
    const { added } = ingestRagRecords(chunkedRecords, seen);
    const totalBytes = chunkedRecords.reduce((total, record) => total + (record.bytes || 0), 0);
    ragTelemetry.lastLoad = Date.now();
    ragTelemetry.lastLoadCount = added;
    ragTelemetry.lastLoadRecords = chunkedRecords.length;
    ragTelemetry.lastLoadBytes = totalBytes;
    ragTelemetry.staticFiles = 0;
    ragTelemetry.staticBytes = 0;
    ragTelemetry.staticErrors = [];
    if (chunkedRecords.length > 0) {
      ragTelemetry.loaded = true;
    }
    renderFloatingMemoryWorkbench();
    updateMemoryStatus();
    updateRagStatusDisplay();

    if (added > 0) {
      const memoryLabel = added === 1 ? 'memory snippet' : 'memory snippets';
      const fileLabel = chunkedRecords.length === 1 ? 'chunked file' : 'chunked files';
      addSystemMessage(`Loaded ${added} ${memoryLabel} from ${chunkedRecords.length} ${fileLabel} into floating memory.`);
    } else if (priorRagCount === countRagMemories()) {
      addSystemMessage('Chunked files were already present in floating memory.');
    } else {
      addSystemMessage('Checked chunked files. No new memories were added.');
    }
  } catch (error) {
    console.error('Failed to load chunked memories', error);
    addSystemMessage('Failed to load chunked memories. Check the debug console for details.');
    void recordLog('error', `Failed to load chunked memories: ${error.message}`, { level: 'error' });
  }
}

async function saveChatTranscriptToArchive() {
  try {
    const messages = await getAllMessages();
    if (!Array.isArray(messages) || messages.length === 0) {
      addSystemMessage('No chat messages available to archive yet.');
      return;
    }

    const sorted = [...messages].sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
    const markdown = formatChatTranscriptMarkdown(sorted);
    const now = new Date();
    const iso = now.toISOString();
    const slug = iso.replace(/[:.]/g, '-');
    const fileName = `chat-${slug}.md`;
    const archivePath = `rag/archives/${fileName}`;
    const label = `Chat ${iso.slice(0, 16).replace('T', ' ')}`;
    const manifestEntry = {
      id: `chat-${slug}`,
      label,
      path: fileName,
      mode: MODE_CHAT,
      role: 'archive',
      format: 'md',
      timestamp: iso
    };
    const archivePayload = {
      content: markdown,
      entry: manifestEntry,
      savedAt: iso,
      messageCount: sorted.length
    };

    cacheArchiveInRam(archivePath, archivePayload);

    let saveLocation = 'workspace';
    try {
      const response = await fetch(archivePath, {
        method: 'PUT',
        headers: { 'Content-Type': 'text/markdown' },
        body: markdown
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    } catch (error) {
      saveLocation = 'ram';
      persistArchiveToStorage(archivePath, archivePayload);
      console.warn('Failed to write chat archive to disk, using RAM disk fallback:', error);
    }

    const manifestLocation = await upsertArchiveManifest(manifestEntry);
    if (manifestLocation === 'ram') {
      saveLocation = 'ram';
    }

    ragTelemetry.lastSave = Date.now();
    ragTelemetry.lastSaveMode = MODE_CHAT;
    ragTelemetry.loaded = true;
    updateRagStatusDisplay();

    const locationLabel = saveLocation === 'ram' ? 'RAM disk' : 'workspace';
    addSystemMessage(`Saved chat transcript to ${archivePath} (${locationLabel}).`);
  } catch (error) {
    console.error('Failed to save chat transcript', error);
    addSystemMessage(`Failed to save chat transcript: ${error.message}`);
    void recordLog('error', `Failed to save chat transcript: ${error.message}`, { level: 'error' });
  }
}

function formatChatTranscriptMarkdown(messages) {
  const lines = ['# Chat transcript', '', `Saved ${new Date().toLocaleString()}`, ''];
  for (const message of messages) {
    const timestamp = new Date(normalizeTimestamp(message.timestamp ?? Date.now())).toISOString();
    const role = (message.role || 'unknown').toUpperCase();
    const turn = message.turnNumber ? ` (turn ${message.turnNumber})` : '';
    lines.push(`## ${role}${turn}`);
    lines.push(`*${timestamp}*`);
    lines.push('');
    lines.push(message.content || '');
    lines.push('');
  }
  return lines.join('\n').trim();
}

async function upsertArchiveManifest(entry) {
  if (!entry) return 'unknown';
  const descriptor = STATIC_RAG_MANIFESTS.find((item) => item.url === 'rag/archives/manifest.json');
  if (!descriptor) return 'unknown';
  const existing = (await fetchRagManifest(descriptor)) || { entries: [] };
  const currentEntries = normalizeRagManifestEntries(existing);
  const nextEntries = currentEntries.filter((item) => {
    if (!item) return false;
    const itemPath = item.path || item.file || item.source;
    if (itemPath && itemPath === entry.path) {
      return false;
    }
    if (item.id && item.id === entry.id) {
      return false;
    }
    return true;
  });
  nextEntries.push(entry);
  const payload = { entries: nextEntries };
  cacheManifestInRam(descriptor, payload);

  try {
    const response = await fetch(descriptor.url, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload, null, 2)
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return 'workspace';
  } catch (error) {
    console.warn('Failed to update archive manifest on disk:', error);
    persistManifestToStorage(descriptor, payload);
    return 'ram';
  }
}

function toChunkedPath(originalPath) {
  if (typeof originalPath !== 'string' || !originalPath) {
    return '';
  }
  const normalized = originalPath.replace(/\\/g, '/');
  if (normalized.startsWith('rag/chunked/')) {
    return normalized;
  }
  if (normalized.startsWith('rag/unchunked/')) {
    return `rag/chunked/${normalized.slice('rag/unchunked/'.length)}`;
  }
  if (normalized.startsWith('rag/')) {
    return `rag/chunked/${normalized.slice('rag/'.length)}`;
  }
  return `rag/chunked/${normalized.replace(/^\//, '')}`;
}

function fromChunkedPath(chunkedPath) {
  if (typeof chunkedPath !== 'string' || !chunkedPath) {
    return '';
  }
  const normalized = chunkedPath.replace(/\\/g, '/');
  if (normalized.startsWith('rag/chunked/')) {
    return `rag/unchunked/${normalized.slice('rag/chunked/'.length)}`;
  }
  return normalized;
}

function toUnchunkedPath(originalPath) {
  if (typeof originalPath !== 'string' || !originalPath) {
    return '';
  }
  const normalized = originalPath.replace(/\\/g, '/');
  if (normalized.startsWith('rag/unchunked/')) {
    return normalized;
  }
  if (normalized.startsWith('rag/chunked/')) {
    return `rag/unchunked/${normalized.slice('rag/chunked/'.length)}`;
  }
  if (normalized.startsWith('rag/')) {
    return `rag/unchunked/${normalized.slice('rag/'.length)}`;
  }
  return `rag/unchunked/${normalized.replace(/^\//, '')}`;
}

function canonicalizeOriginalPath(path) {
  if (typeof path !== 'string' || !path) {
    return '';
  }
  return toUnchunkedPath(path);
}

function dedupePaths(paths) {
  return Array.from(new Set((Array.isArray(paths) ? paths : [paths]).filter(Boolean)));
}

function cacheChunkedInRam(paths, data) {
  if (!data) return;
  const targets = new Set(dedupePaths(paths));
  if (data && typeof data.originalPath === 'string') {
    targets.add(data.originalPath);
    targets.add(toChunkedPath(data.originalPath));
  }
  if (data && typeof data.chunkedPath === 'string') {
    targets.add(data.chunkedPath);
  }
  for (const path of targets) {
    if (path) {
      ramDiskCache.chunked.set(path, data);
    }
  }
}

function cacheUnchunkedInRam(paths, content) {
  if (typeof content !== 'string') return;
  const targets = dedupePaths(paths);
  for (const path of targets) {
    if (path) {
      ramDiskCache.unchunked.set(path, content);
    }
  }
}

function clearUnchunkedRamEntries(paths) {
  const targets = dedupePaths(paths);
  for (const path of targets) {
    if (!path) continue;
    ramDiskCache.unchunked.delete(path);
    try {
      storageRemoveItem(`${RAM_DISK_UNCHUNKED_PREFIX}${path}`);
    } catch (error) {
      console.debug(`Failed to remove cached unchunked entry for ${path}:`, error);
    }
  }
}

function cacheArchiveInRam(path, payload) {
  if (!path || !payload) return;
  ramDiskCache.archives.set(path, payload);
}

function persistArchiveToStorage(path, payload) {
  if (!path || !payload) return;
  try {
    storageSetItem(`${RAM_DISK_ARCHIVE_PREFIX}${path}`, JSON.stringify(payload));
  } catch (error) {
    console.warn(`Failed to persist archive ${path} to RAM disk storage:`, error);
  }
}

function listRamDiskArchiveEntries() {
  const results = new Map();
  for (const [path, data] of ramDiskCache.archives.entries()) {
    results.set(path, { path, data });
  }
  const prefixLength = RAM_DISK_ARCHIVE_PREFIX.length;
  for (const key of storageKeys()) {
    if (!key.startsWith(RAM_DISK_ARCHIVE_PREFIX)) continue;
    const path = key.slice(prefixLength);
    if (!path || results.has(path)) continue;
    try {
      const stored = storageGetItem(key);
      if (!stored) continue;
      const parsed = JSON.parse(stored);
      if (!parsed) continue;
      ramDiskCache.archives.set(path, parsed);
      results.set(path, { path, data: parsed });
    } catch (error) {
      console.warn(`Failed to read archived transcript from RAM disk (${key}):`, error);
    }
  }
  return Array.from(results.values());
}

function readRamDiskChunk(paths) {
  const candidates = dedupePaths(paths);
  for (const path of candidates) {
    if (ramDiskCache.chunked.has(path)) {
      return ramDiskCache.chunked.get(path);
    }
  }
  for (const path of candidates) {
    const stored = storageGetItem(`${RAM_DISK_CHUNK_PREFIX}${path}`);
    if (!stored) continue;
    try {
      const data = JSON.parse(stored);
      if (typeof data.chunkedPath !== 'string') {
        data.chunkedPath = path;
      }
      cacheChunkedInRam(candidates, data);
      return data;
    } catch (error) {
      console.warn(`Failed to parse RAM disk chunk for ${path}:`, error);
    }
  }
  return null;
}

function readRamDiskUnchunked(paths) {
  const candidates = dedupePaths(paths);
  for (const path of candidates) {
    if (ramDiskCache.unchunked.has(path)) {
      return ramDiskCache.unchunked.get(path);
    }
  }
  for (const path of candidates) {
    const stored = storageGetItem(`${RAM_DISK_UNCHUNKED_PREFIX}${path}`);
    if (stored !== null && stored !== undefined) {
      cacheUnchunkedInRam(candidates, stored);
      return stored;
    }
  }
  return null;
}

function listRamDiskChunkedEntries() {
  const entries = [];
  const keys = storageKeys().filter((key) => key.startsWith(RAM_DISK_CHUNK_PREFIX));
  for (const key of keys) {
    const path = key.slice(RAM_DISK_CHUNK_PREFIX.length);
    const data = readRamDiskChunk(path);
    if (data && isSupportedChunkablePath(path)) {
      entries.push({ path, data });
    }
  }
  return entries;
}

function listRamDiskUnchunkedEntries() {
  const entries = [];
  const keys = storageKeys().filter((key) => key.startsWith(RAM_DISK_UNCHUNKED_PREFIX));
  for (const key of keys) {
    const path = key.slice(RAM_DISK_UNCHUNKED_PREFIX.length);
    const content = readRamDiskUnchunked(path);
    if (content !== null && isSupportedChunkablePath(path)) {
      entries.push({ path, content });
    }
  }
  return entries;
}

function primeRamDiskCache() {
  try {
    const keys = storageKeys();
    for (const key of keys) {
      if (key.startsWith(RAM_DISK_CHUNK_PREFIX)) {
        const path = key.slice(RAM_DISK_CHUNK_PREFIX.length);
        const stored = storageGetItem(key);
        if (!stored) continue;
        try {
          const data = JSON.parse(stored);
          cacheChunkedInRam([path], data);
        } catch (error) {
          console.warn(`Failed to parse cached chunked entry ${path}:`, error);
        }
      } else if (key.startsWith(RAM_DISK_UNCHUNKED_PREFIX)) {
        const path = key.slice(RAM_DISK_UNCHUNKED_PREFIX.length);
        const stored = storageGetItem(key);
        if (stored !== null && stored !== undefined) {
          cacheUnchunkedInRam([path], stored);
        }
      }
    }
  } catch (error) {
    console.warn('Failed to prime RAM disk cache:', error);
  }
}

function normalizeRagData(data) {
  if (data == null) {
    return '';
  }
  if (typeof data === 'string') {
    return data;
  }
  if (Array.isArray(data)) {
    return data.map((item) => normalizeRagData(item)).join('\n');
  }
  if (typeof data === 'object') {
    if (typeof data.content === 'string') {
      return data.content;
    }
    return JSON.stringify(data);
  }
  return String(data);
}

function normalizeRagString(raw) {
  if (typeof raw !== 'string') {
    return normalizeRagData(raw);
  }
  const trimmed = raw.trim();
  if (!trimmed) {
    return '';
  }
  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    try {
      const parsed = JSON.parse(trimmed);
      return normalizeRagData(parsed);
    } catch (error) {
      const lines = trimmed.split(/\r?\n/);
      const parsedLines = [];
      let parsedCount = 0;
      for (const line of lines) {
        const candidate = line.trim();
        if (!candidate) continue;
        if (candidate.startsWith('{') || candidate.startsWith('[')) {
          try {
            const parsed = JSON.parse(candidate);
            parsedLines.push(normalizeRagData(parsed));
            parsedCount += 1;
            continue;
          } catch (lineError) {
            // fall through to keep the original line
          }
        }
        parsedLines.push(line);
      }
      if (parsedCount > 0) {
        return parsedLines.join('\n');
      }
    }
  }
  return raw;
}

async function loadChunkedFileData(chunkedPath, originalPath) {
  const candidates = dedupePaths([chunkedPath, originalPath]);
  const cached = readRamDiskChunk(candidates);
  if (cached) {
    return { data: cached, source: 'ram' };
  }
  for (const path of candidates) {
    const fsResult = readStandaloneFile(path);
    if (fsResult) {
      try {
        const payload = JSON.parse(fsResult.content);
        const normalized = { ...payload, chunkedPath: path };
        hydrateChunkEmbeddingIndex(normalized);
        cacheChunkedInRam(candidates, normalized);
        return { data: normalized, source: 'filesystem' };
      } catch (error) {
        console.debug(`Failed to parse chunked file ${path} from filesystem:`, error);
      }
    }
    try {
      const response = await fetch(path);
      if (response.ok) {
        const payload = await response.json();
        const normalized = { ...payload, chunkedPath: path };
        hydrateChunkEmbeddingIndex(normalized);
        cacheChunkedInRam(candidates, normalized);
        return { data: normalized, source: 'filesystem' };
      }
    } catch (error) {
      console.debug(`Failed to load chunked file ${path}:`, error);
    }
  }
  return null;
}

async function loadRagFileContent(paths) {
  const candidates = dedupePaths(paths);
  for (const path of candidates) {
    const fsResult = readStandaloneFile(path);
    if (fsResult) {
      const { content, contentType } = fsResult;
      if (contentType === 'application/pdf') {
        // Defer to fetch/pdf.js pipeline for binary sources.
      } else if (contentType && contentType.includes('json')) {
        try {
          const parsed = JSON.parse(content);
          const raw = typeof parsed === 'string' ? parsed : JSON.stringify(parsed);
          cacheUnchunkedInRam(candidates, raw);
          return { content: normalizeRagData(parsed), source: 'filesystem', contentType: 'application/json' };
        } catch (error) {
          cacheUnchunkedInRam(candidates, content);
          return { content, source: 'filesystem', contentType: 'application/json' };
        }
      } else {
        cacheUnchunkedInRam(candidates, content);
        return { content, source: 'filesystem', contentType: contentType || 'text/plain' };
      }
    }
    try {
      const response = await fetch(path);
      if (response.ok) {
        const contentType = response.headers.get('content-type') || '';
        if (contentType.includes('application/json')) {
          const data = await response.json();
          const raw = typeof data === 'string' ? data : JSON.stringify(data);
          cacheUnchunkedInRam(candidates, raw);
          return { content: normalizeRagData(data), source: 'filesystem', contentType };
        }
        const text = await response.text();
        cacheUnchunkedInRam(candidates, text);
        return { content: text, source: 'filesystem', contentType: contentType || 'text/plain' };
      }
    } catch (error) {
      console.debug(`Failed to fetch ${path} for chunking`, error);
    }
  }
  const ramContent = readRamDiskUnchunked(candidates);
  if (ramContent !== null) {
    return { content: normalizeRagString(ramContent), source: 'ram', contentType: 'text/plain' };
  }
  return null;
}

async function detectUnchunkedFile(originalPath, manifestSize = 0) {
  const candidatePaths = dedupePaths([toUnchunkedPath(originalPath), originalPath]);
  for (const path of candidatePaths) {
    const fsResult = readStandaloneFile(path);
    if (fsResult) {
      const { content, contentType } = fsResult;
      if (!isSupportedChunkableFile(originalPath, contentType)) {
        continue;
      }
      cacheUnchunkedInRam(candidatePaths, content);
      const computedSize = manifestSize || content.length;
      return {
        path,
        originalPath,
        size: computedSize,
        source: 'workspace',
        ramContent: content,
        contentType: contentType || inferContentTypeFromPath(originalPath)
      };
    }
    try {
      const response = await fetch(path);
      if (response.ok) {
        const contentType = response.headers.get('content-type') || '';
        if (!isSupportedChunkableFile(originalPath, contentType)) {
          continue;
        }
        const text = await response.text();
        cacheUnchunkedInRam(candidatePaths, text);
        const sizeHeader = Number.parseInt(response.headers.get('content-length') || '', 10);
        const computedSize = Number.isFinite(sizeHeader) ? sizeHeader : manifestSize || text.length;
        return {
          path,
          originalPath,
          size: computedSize,
          source: 'workspace',
          ramContent: text,
          contentType
        };
      }
    } catch (error) {
      console.debug(`Failed to inspect ${path}:`, error);
    }
  }
  const ramContent = readRamDiskUnchunked(candidatePaths);
  if (ramContent !== null) {
    if (!isSupportedChunkableFile(originalPath, null)) {
      return null;
    }
    return {
      path: toUnchunkedPath(originalPath),
      originalPath,
      size: manifestSize || ramContent.length,
      source: 'ram',
      ramContent
    };
  }
  return null;
}

async function listRagDirectoryEntries(directory) {
  const normalizedDir = directory.endsWith('/') ? directory : `${directory}/`;
  const fsEntries = listStandaloneDirectory(normalizedDir);
  if (Array.isArray(fsEntries)) {
    return filterSuspiciousDirectoryEntries(fsEntries);
  }
  try {
    const response = await fetch(normalizedDir, { cache: 'no-store' });
    if (!response.ok) {
      return [];
    }
    const contentType = (response.headers.get('content-type') || '').toLowerCase();
    if (contentType.includes('application/json')) {
      const payload = await response.json();
      return normalizeDirectoryListingFromJson(payload, normalizedDir);
    }
    const text = await response.text();
    return normalizeDirectoryListingFromText(text, normalizedDir);
  } catch (error) {
    console.debug(`Failed to list directory ${directory}:`, error);
    return [];
  }
}

function normalizeDirectoryListingFromJson(payload, basePath) {
  const results = new Set();
  if (!payload) return [];
  const base = basePath.endsWith('/') ? basePath : `${basePath}/`;
  const candidateArrays = [];
  if (Array.isArray(payload)) {
    candidateArrays.push(payload);
  }
  if (Array.isArray(payload.files)) {
    candidateArrays.push(payload.files);
  }
  if (Array.isArray(payload.entries)) {
    candidateArrays.push(payload.entries);
  }
  for (const array of candidateArrays) {
    for (const item of array) {
      if (!item) continue;
      if (typeof item === 'string') {
        results.add(resolveRagPath(base, item));
      } else if (typeof item.path === 'string') {
        results.add(resolveRagPath(base, item.path));
      }
    }
  }
  return filterSuspiciousDirectoryEntries(Array.from(results));
}

function normalizeDirectoryListingFromText(text, basePath) {
  const results = new Set();
  if (!text) return [];
  const normalizedBase = basePath.endsWith('/') ? basePath : `${basePath}/`;
  try {
    const parser = new DOMParser();
    const baseUrl = new URL(normalizedBase, window.location.origin);
    const expectedPrefix = baseUrl.pathname.replace(/^\//, '');
    const doc = parser.parseFromString(text, 'text/html');
    const anchors = Array.from(doc.querySelectorAll('a[href]'));
    for (const anchor of anchors) {
      const href = anchor.getAttribute('href');
      if (!href || href.startsWith('#') || href.startsWith('?')) continue;
      const resolvedUrl = new URL(href, baseUrl);
      const pathname = resolvedUrl.pathname.replace(/^\//, '');
      if (!pathname || pathname === expectedPrefix) continue;
      if (!pathname.startsWith(expectedPrefix)) continue;
      if (pathname.endsWith('/')) continue;
      if (/["']/.test(pathname)) continue;
      results.add(pathname);
    }
  } catch (error) {
    console.debug('Failed to parse directory listing HTML', error);
  }

  if (results.size === 0) {
    const lines = text.split(/\r?\n/);
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed === '.' || trimmed === '..') continue;
      if (/[<>]/.test(trimmed)) continue;
      if (/["']/.test(trimmed) && /=/.test(trimmed)) continue;
      if (trimmed.endsWith('/')) continue;
      results.add(resolveRagPath(normalizedBase, trimmed));
    }
  }

  return filterSuspiciousDirectoryEntries(Array.from(results));
}

function filterSuspiciousDirectoryEntries(entries) {
  if (!Array.isArray(entries)) return [];
  return entries.filter((entry) => {
    if (!entry) return false;
    if (/["'<>]/.test(entry)) return false;
    const lastSegment = entry.split('/').pop();
    if (!lastSegment || lastSegment === '.' || lastSegment === '..') return false;
    if (/["'<>]/.test(lastSegment)) return false;
    return true;
  });
}

async function scanForChunkableFiles() {
  if (chunkerState.isProcessing) {
    addSystemMessage('Chunker is already processing. Please wait.');
    return;
  }

  try {
    chunkerState.isProcessing = true;
    updateChunkerStatus();

    addSystemMessage('🔍 Scanning rag/ folder for chunkable files...');

    const [manifest, unchunkedListing, chunkedListing] = await Promise.all([
      fetchRagManifest(STATIC_RAG_MANIFESTS[0]).catch((error) => {
        console.warn('Could not load primary RAG manifest for chunking', error);
        return null;
      }),
      listRagDirectoryEntries('rag/unchunked/'),
      listRagDirectoryEntries('rag/chunked/')
    ]);

    chunkerState.unchunkedFiles = [];
    chunkerState.chunkedFiles = [];
    chunkerState.totalChunks = 0;

    const seenChunked = new Set();
    const seenUnchunked = new Set();

    const chunkedPaths = dedupePaths(chunkedListing);
    for (const chunkedPath of chunkedPaths) {
      try {
        const chunkedRecord = await loadChunkedFileData(chunkedPath, fromChunkedPath(chunkedPath));
        const chunkList = chunkedRecord?.data?.chunks;
        if (!Array.isArray(chunkList) || chunkList.length === 0) {
          continue;
        }
        const originalPath = chunkedRecord?.data?.originalPath || fromChunkedPath(chunkedPath);
        const canonical = canonicalizeOriginalPath(originalPath || chunkedPath);
        if (seenChunked.has(canonical)) {
          continue;
        }
        chunkerState.chunkedFiles.push({
          originalPath,
          chunkedPath,
          chunks: chunkList,
          source: chunkedRecord.source
        });
        chunkerState.totalChunks += chunkList.length;
        seenChunked.add(canonical);
        const label = (originalPath || chunkedPath).replace(/^rag\//, '');
        const locationLabel = chunkedRecord.source === 'ram' ? 'RAM disk' : 'workspace';
        addSystemMessage(`✓ Found chunked: ${label} (${chunkList.length} chunks, ${locationLabel})`);
      } catch (error) {
        console.debug(`Failed to inspect chunked file ${chunkedPath}:`, error);
      }
    }

    const manifestEntries = normalizeRagManifestEntries(manifest);
    const manifestPaths = manifestEntries
      .map((entry) => (entry && entry.path ? resolveRagPath('rag/', entry.path) : null))
      .filter(Boolean);

    const candidatePaths = filterChunkablePaths(dedupePaths([...manifestPaths, ...unchunkedListing]));

    for (const candidate of candidatePaths) {
      const canonicalCandidate = canonicalizeOriginalPath(candidate);
      if (seenChunked.has(canonicalCandidate) || seenUnchunked.has(canonicalCandidate)) {
        continue;
      }

      let chunkedRecord = null;
      try {
        const chunkedPath = toChunkedPath(candidate);
        chunkedRecord = await loadChunkedFileData(chunkedPath, candidate);
      } catch (error) {
        console.debug(`Chunked probe failed for ${candidate}:`, error);
      }

      const chunkList = chunkedRecord?.data?.chunks;
      if (Array.isArray(chunkList) && chunkList.length > 0) {
        const originalPath = chunkedRecord?.data?.originalPath || candidate;
        const canonicalOriginal = canonicalizeOriginalPath(originalPath);
        if (!seenChunked.has(canonicalOriginal)) {
          chunkerState.chunkedFiles.push({
            originalPath,
            chunkedPath: toChunkedPath(originalPath),
            chunks: chunkList,
            source: chunkedRecord.source
          });
          chunkerState.totalChunks += chunkList.length;
          seenChunked.add(canonicalOriginal);
          const label = originalPath.replace(/^rag\//, '');
          const locationLabel = chunkedRecord.source === 'ram' ? 'RAM disk' : 'workspace';
          addSystemMessage(`✓ Found chunked: ${label} (${chunkList.length} chunks, ${locationLabel})`);
        }
        continue;
      }

      const descriptor = await detectUnchunkedFile(candidate);
      if (descriptor) {
        const descriptorKey = canonicalizeOriginalPath(descriptor.originalPath || descriptor.path);
        if (seenChunked.has(descriptorKey) || seenUnchunked.has(descriptorKey)) {
          continue;
        }
        if (!isSupportedChunkablePath(descriptor.originalPath || descriptor.path)) {
          continue;
        }
        chunkerState.unchunkedFiles.push(descriptor);
        seenUnchunked.add(descriptorKey);
        const locationLabel = descriptor.source === 'ram' ? 'RAM disk' : 'workspace';
        const icon = descriptor.source === 'ram' ? '📦' : '📁';
        const label = (descriptor.originalPath || descriptor.path).replace(/^rag\//, '');
        addSystemMessage(`${icon} Found unchunked: ${label} (${locationLabel})`);
      }
    }

    for (const { path, data } of listRamDiskChunkedEntries()) {
      const originalPath = typeof data?.originalPath === 'string' ? data.originalPath : fromChunkedPath(path);
      const canonical = canonicalizeOriginalPath(originalPath || path);
      if (!originalPath || seenChunked.has(canonical) || !isSupportedChunkablePath(originalPath)) {
        continue;
      }
      const chunks = Array.isArray(data?.chunks) ? data.chunks : [];
      if (!chunks.length) {
        continue;
      }
      chunkerState.chunkedFiles.push({
        originalPath,
        chunkedPath: toChunkedPath(originalPath),
        chunks,
        source: 'ram'
      });
      chunkerState.totalChunks += chunks.length;
      seenChunked.add(canonical);
      const label = originalPath.startsWith('rag/') ? originalPath.slice(4) : originalPath;
      addSystemMessage(`✓ Found chunked: ${label} (${chunks.length} chunks, RAM disk)`);
    }

    for (const { path, content } of listRamDiskUnchunkedEntries()) {
      const canonical = canonicalizeOriginalPath(path);
      if (!path || seenChunked.has(canonical) || seenUnchunked.has(canonical) || !isSupportedChunkablePath(path)) {
        continue;
      }
      const descriptor = {
        path: toUnchunkedPath(path),
        originalPath: path,
        size: content.length,
        source: 'ram',
        ramContent: content
      };
      chunkerState.unchunkedFiles.push(descriptor);
      seenUnchunked.add(canonical);
      const label = path.startsWith('rag/') ? path.slice(4) : path;
      addSystemMessage(`📦 Found unchunked: ${label} (RAM disk)`);
    }

    updateChunkerStatus();
    addSystemMessage(
      `✅ Scan complete. Found ${chunkerState.unchunkedFiles.length} files to chunk, ${chunkerState.chunkedFiles.length} already chunked.`
    );

  } catch (error) {
    console.error('Error scanning for chunkable files:', error);
    addSystemMessage(`❌ Scan failed: ${error.message || 'Unknown error'}`);
    void recordLog('error', `Scan for chunkable files failed: ${error.message || 'Unknown error'}`, { level: 'error' });
  } finally {
    chunkerState.isProcessing = false;
    updateChunkerStatus();
  }
}

async function chunkAllFiles() {
  if (chunkerState.isProcessing) {
    addSystemMessage('Chunker is already processing. Please wait.');
    return;
  }

  if (chunkerState.unchunkedFiles.length === 0) {
    addSystemMessage('No files to chunk. Scan for files first.');
    return;
  }

  try {
    chunkerState.isProcessing = true;
    updateChunkerStatus();

    let processedCount = 0;
    let totalChunksCreated = 0;

    for (const file of chunkerState.unchunkedFiles) {
      try {
        const originalPath = file.originalPath || file.path;
        const displayPath = originalPath || file.path;
        if (!isSupportedChunkablePath(originalPath || '')) {
          addSystemMessage(`Skipping ${displayPath}: unsupported file type for chunking.`);
          continue;
        }
        addSystemMessage(`Chunking ${displayPath}...`);
        let chunks = await chunkFile(file, chunkerState.chunkSize, chunkerState.chunkOverlap);
        chunks = await enrichChunksWithEmbeddings(originalPath, chunks);
        if (Array.isArray(chunks) && chunks.length > 0) {
          const chunkedData = {
            originalPath,
            chunkSize: chunkerState.chunkSize,
            chunkOverlap: chunkerState.chunkOverlap,
            embeddingModel: config.embeddingModel,
            embeddingDimensions: Array.isArray(chunks[0]?.embedding)
              ? chunks[0].embedding.length
              : EMBEDDING_HASH_BUCKETS,
            chunks,
            createdAt: new Date().toISOString()
          };

          // Save chunked file to rag/chunked/
          const chunkedPath = toChunkedPath(originalPath);
          const saveLocation = await saveChunkedFileToFS(chunkedPath, chunkedData);

          // Move original to rag/unchunked/
          await moveToUnchunkedFS(file.path, originalPath);

          chunkerState.chunkedFiles.push({
            originalPath,
            chunkedPath,
            chunks,
            source: saveLocation
          });

          totalChunksCreated += chunks.length;
          processedCount++;
          const locationLabel = saveLocation === 'ram' ? 'RAM disk' : 'workspace';
          addSystemMessage(`✓ Chunked ${displayPath} into ${chunks.length} chunks (${locationLabel})`);
        }
      } catch (error) {
        const originalPath = file.originalPath || file.path;
        console.error(`Error chunking ${originalPath}:`, error);
        addSystemMessage(`✗ Failed to chunk ${originalPath}: ${error.message || 'Unknown error'}`);
      }
    }

    chunkerState.unchunkedFiles = [];
    chunkerState.totalChunks += totalChunksCreated;

    updateChunkerStatus();
    addSystemMessage(`✓ Chunking complete. Processed ${processedCount} files, created ${totalChunksCreated} chunks.`);

  } catch (error) {
    console.error('Error during chunking:', error);
    addSystemMessage(`✗ Chunking failed: ${error.message || 'Unknown error'}`);
    void recordLog('error', `Chunking failed: ${error.message || 'Unknown error'}`, { level: 'error' });
  } finally {
    chunkerState.isProcessing = false;
    updateChunkerStatus();
  }
}

async function chunkFile(fileDescriptor, chunkSize, overlap) {
  const descriptor =
    typeof fileDescriptor === 'string'
      ? { path: fileDescriptor, originalPath: fileDescriptor }
      : { ...fileDescriptor };
  const label = descriptor.originalPath || descriptor.path || 'unknown file';
  const candidatePaths = dedupePaths([descriptor.path, descriptor.originalPath]);
  const overlapTokens = Math.max(Math.round(chunkSize * 0.15), overlap);

  try {
    let sourceContent = '';

    if (typeof descriptor.ramContent === 'string') {
      cacheUnchunkedInRam(candidatePaths, descriptor.ramContent);
      sourceContent = normalizeRagString(descriptor.ramContent);
    } else {
      const loaded = await loadRagFileContent(candidatePaths);
      if (loaded && typeof loaded.content === 'string') {
        sourceContent = normalizeRagString(loaded.content);
      } else {
        const fallback = readRamDiskUnchunked(candidatePaths);
        if (fallback !== null) {
          sourceContent = normalizeRagString(fallback);
        } else {
          throw new Error('File not found in workspace or RAM disk');
        }
      }
    }

    if (!sourceContent.trim()) {
      return [];
    }

    const segments = sourceContent.split(/\n\s*\n/).filter((segment) => segment.trim());
    const chunks = [];
    let currentChunk = '';
    let currentTokens = 0;

    for (const segment of segments) {
      const segmentTokens = estimateTokenCount(segment);

      if (currentTokens + segmentTokens > chunkSize && currentChunk.trim()) {
        chunks.push({
          content: currentChunk.trim(),
          tokens: currentTokens,
          startIndex: chunks.length * Math.max(1, chunkSize - overlapTokens)
        });

        const overlapText = extractOverlapText(currentChunk, overlapTokens);
        currentChunk = overlapText ? overlapText + segment : segment;
        currentTokens = estimateTokenCount(currentChunk);
      } else {
        currentChunk += (currentChunk ? '\n\n' : '') + segment;
        currentTokens += segmentTokens;
      }
    }

    if (currentChunk.trim()) {
      chunks.push({
        content: currentChunk.trim(),
        tokens: currentTokens,
        startIndex: chunks.length * Math.max(1, chunkSize - overlapTokens)
      });
    }

    return chunks;
  } catch (error) {
    console.error(`Error chunking file ${label}:`, error);
    void recordLog('error', `Error chunking file ${label}: ${error.message}`, { level: 'error' });
    throw error;
  }
}

function extractOverlapText(text, overlapTokens) {
  const sentences = text.split(/[.!?]+/).filter(s => s.trim());
  let overlapText = '';
  let tokens = 0;

  for (let i = sentences.length - 1; i >= 0; i--) {
    const sentence = sentences[i].trim();
    if (!sentence) continue;

    const sentenceTokens = estimateTokenCount(sentence);
    if (tokens + sentenceTokens > overlapTokens) {
      break;
    }

    overlapText = sentence + (overlapText ? '. ' : '') + overlapText;
    tokens += sentenceTokens;
  }

  return overlapText;
}

// CODEx: Populate chunk embeddings so RAG retrieval can perform vector search.
async function enrichChunksWithEmbeddings(originalPath, chunks) {
  if (!Array.isArray(chunks) || !chunks.length) {
    return chunks;
  }
  const canonical = canonicalizeOriginalPath(originalPath || '');
  const enriched = await embedAndStore(chunks, {
    sourceId: canonical,
    embedder: (content) => generateTextEmbedding(content), // CODEx: Request provider embeddings for each chunk body.
    lexicalEmbedder: (content) => lexicalEmbedding(content), // CODEx: Produce deterministic fallback vectors when provider fails.
    registerEmbedding: (sourceId, index, vector) => {
      storeChunkEmbedding(`${sourceId}#${index}`, vector); // CODEx: Cache embeddings with footprint tracking per chunk.
    }
  });
  return enriched; // CODEx: Return enriched chunk array containing embeddings for persistence.
}

// CODEx: Cache embeddings when chunk files are reloaded from disk or RAM.
function hydrateChunkEmbeddingIndex(record) {
  if (!record || !Array.isArray(record.chunks)) {
    return;
  }
  const canonical = canonicalizeOriginalPath(record.originalPath || record.chunkedPath || '');
  record.chunks.forEach((chunk, index) => {
    if (chunk && Array.isArray(chunk.embedding) && chunk.embedding.length) {
      const key = `${canonical}#${index}`;
      storeChunkEmbedding(key, chunk.embedding); // CODEx: Rehydrate embedding cache while preserving footprint accounting.
    }
  });
}

async function saveChunkedFileToFS(path, data) {
  const payloadForCache = { ...data, chunkedPath: path };
  cacheChunkedInRam([path, data?.originalPath], payloadForCache);
  const serialized = JSON.stringify(data, null, 2);
  const directory = getDirectoryPath(path);
  if (directory) {
    ensureStandaloneDirectory(directory);
  }
  if (writeStandaloneFile(path, serialized, { contentType: 'application/json' })) {
    console.log(`Saved chunked file to ${path} via standalone bridge`);
    return 'workspace';
  }
  try {
    const response = await fetch(path, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: serialized
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    console.log(`Saved chunked file to ${path}`);
    return 'workspace';
  } catch (error) {
    console.error('Error saving chunked file to FS:', error);
    try {
      storageSetItem(`${RAM_DISK_CHUNK_PREFIX}${path}`, JSON.stringify(payloadForCache));
      cacheChunkedInRam([path, data?.originalPath], payloadForCache);
      console.log(`Fallback: Saved chunked file to localStorage: ${RAM_DISK_CHUNK_PREFIX}${path}`);
    } catch (storageError) {
      console.warn('Failed to persist chunked file to localStorage:', storageError);
    }
    return 'ram';
  }
}

async function moveToUnchunkedFS(fromPath, originalPath) {
  const normalizedSource = (originalPath || fromPath || '').replace(/\\/g, '/');
  if (!normalizedSource) {
    return 'unknown';
  }
  const unchunkedPath = toUnchunkedPath(normalizedSource);
  const candidatePaths = dedupePaths([fromPath, normalizedSource, unchunkedPath]);

  if (normalizedSource.startsWith('rag/unchunked/')) {
    if (deleteStandaloneFile(normalizedSource)) {
      clearUnchunkedRamEntries(candidatePaths);
      console.log(`Deleted ${normalizedSource} via standalone bridge`);
      return 'workspace';
    }
    const cached = readRamDiskUnchunked(candidatePaths);
    try {
      const deleteResponse = await fetch(normalizedSource, { method: 'DELETE' });
      if (deleteResponse?.ok) {
        clearUnchunkedRamEntries(candidatePaths);
        console.log(`Deleted ${normalizedSource} after chunking`);
        return 'workspace';
      }
    } catch (error) {
      console.warn(`Could not delete processed unchunked file ${normalizedSource}:`, error);
    }

    if (typeof cached === 'string') {
      cacheUnchunkedInRam(candidatePaths, cached);
      for (const target of candidatePaths) {
        if (!target) continue;
        try {
          storageSetItem(`${RAM_DISK_UNCHUNKED_PREFIX}${target}`, cached);
        } catch (storageError) {
          console.warn(`Failed to persist unchunked fallback for ${target}:`, storageError);
        }
      }
      console.log(`Fallback: retained ${normalizedSource} in RAM disk storage`);
      return 'ram';
    }
    return 'unknown';
  }

  const fsSource = readStandaloneFile(fromPath) || readStandaloneFile(normalizedSource);
  if (fsSource && typeof fsSource.content === 'string') {
    ensureStandaloneDirectory(getDirectoryPath(unchunkedPath));
    if (writeStandaloneFile(unchunkedPath, fsSource.content, { contentType: fsSource.contentType })) {
      deleteStandaloneFile(fromPath);
      cacheUnchunkedInRam(candidatePaths, fsSource.content);
      console.log(`Moved ${fromPath} to ${unchunkedPath} via standalone bridge`);
      return 'workspace';
    }
  }

  try {
    const response = await fetch(fromPath);
    if (response.ok) {
      const content = await response.text();
      const moveResponse = await fetch(unchunkedPath, {
        method: 'PUT',
        headers: { 'Content-Type': 'text/plain' },
        body: content
      });
      if (moveResponse.ok) {
        await fetch(fromPath, { method: 'DELETE' });
        cacheUnchunkedInRam(candidatePaths, content);
        console.log(`Moved ${fromPath} to ${unchunkedPath}`);
        return 'workspace';
      }
    }
  } catch (error) {
    console.warn('Could not move file to unchunked folder, using localStorage fallback:', error);
  }

  try {
    let content = readRamDiskUnchunked(candidatePaths);
    if (content === null) {
      const response = await fetch(fromPath);
      if (response.ok) {
        content = await response.text();
      }
    }
    if (typeof content === 'string') {
      cacheUnchunkedInRam(candidatePaths, content);
      for (const target of candidatePaths) {
        if (!target) continue;
        try {
          storageSetItem(`${RAM_DISK_UNCHUNKED_PREFIX}${target}`, content);
        } catch (storageError) {
          console.warn(`Failed to persist unchunked file to localStorage (${target}):`, storageError);
        }
      }
      console.log(`Fallback: Moved file to RAM disk storage for ${normalizedSource}`);
      return 'ram';
    }
  } catch (fallbackError) {
    console.warn('Fallback also failed:', fallbackError);
  }

  return 'unknown';
}

function clearAllChunks() {
  try {
    addSystemMessage('🗑️ Clearing all chunked and unchunked files...');

    // Clear localStorage entries
    const keys = storageKeys().filter(
      (key) => key.startsWith(RAM_DISK_CHUNK_PREFIX) || key.startsWith(RAM_DISK_UNCHUNKED_PREFIX)
    );
    keys.forEach(key => storageRemoveItem(key));

    // Try to delete files from file system
    const deletePromises = [];
    chunkerState.chunkedFiles.forEach(file => {
      if (file.chunkedPath) {
        if (deleteStandaloneFile(file.chunkedPath)) {
          addSystemMessage(`✓ Deleted ${file.chunkedPath}`);
          return deletePromises.push(Promise.resolve());
        }
        deletePromises.push(
          fetch(file.chunkedPath, { method: 'DELETE' })
            .then(() => addSystemMessage(`✓ Deleted ${file.chunkedPath}`))
            .catch(() => addSystemMessage(`⚠️ Could not delete ${file.chunkedPath}`))
        );
      }
    });
    chunkerState.unchunkedFiles.forEach(file => {
      if (file.path && file.path.includes('/unchunked/')) {
        if (deleteStandaloneFile(file.path)) {
          addSystemMessage(`✓ Deleted ${file.path}`);
          return deletePromises.push(Promise.resolve());
        }
        deletePromises.push(
          fetch(file.path, { method: 'DELETE' })
            .then(() => addSystemMessage(`✓ Deleted ${file.path}`))
            .catch(() => addSystemMessage(`⚠️ Could not delete ${file.path}`))
        );
      }
    });

    // Wait for all delete operations
    Promise.allSettled(deletePromises).then(() => {
      chunkerState.unchunkedFiles = [];
      chunkerState.chunkedFiles = [];
      chunkerState.totalChunks = 0;
      ramDiskCache.chunked.clear();
      ramDiskCache.unchunked.clear();
      updateChunkerStatus();
      addSystemMessage(`✅ Cleared all chunked and unchunked files.`);
    });

  } catch (error) {
    console.error('Error clearing chunks:', error);
    addSystemMessage(`❌ Failed to clear chunks: ${error.message || 'Unknown error'}`);
    void recordLog('error', `Failed to clear chunks: ${error.message || 'Unknown error'}`, { level: 'error' });
  }
}

async function handleUserMessage() {
  const rawMessage = elements.messageInput.value.trim();
  if (!rawMessage) return;

  elements.messageInput.value = '';
  const entry = await appendMessage('user', rawMessage, { speak: false });
  await respondToUser(entry);
}

async function respondToUser(userEntry) {
  const autoInjecting = Boolean(config.autoInjectMemories);
  updateProcessState('modelA', {
    status: 'Retrieving',
    detail: autoInjecting ? 'Gathering relevant memories.' : 'Auto-injection disabled; using floating buffer as needed.'
  });
  try {
    const { messages, retrievedMemories } = await buildModelMessages(userEntry);
    if (autoInjecting) {
      registerRetrieval('A', retrievedMemories.length);
    }
    if (!config.endpoint || !config.model) {
      addSystemMessage('Configure the model endpoint and name to receive AI replies.');
      updateModelConnectionStatus();
      updateProcessState('modelA', { status: 'Idle', detail: 'Waiting for user prompt.' });
      return;
    }
    updateModelStatus('Contacting model…', 'status-pill--idle');
    updateProcessState('modelA', {
      status: 'Contacting model',
      detail: `Calling ${config.model || 'configured model'} at ${config.endpoint}`
    });
    modelAInFlight = true;
    const { content: response, truncated, reasoning } = await callModel(messages);
    if (!response) {
      updateModelStatus('Model returned empty response', 'status-pill--idle');
      updateProcessState('modelA', { status: 'Idle', detail: 'Model returned empty response.' });
      setTimeout(() => updateModelConnectionStatus(), 1500);
      return;
    }
    updateModelStatus('Model response received', 'status-pill--ready');
    if (truncated) {
      addSystemMessage(
        `Assistant reply trimmed to stay within the ~${config.maxResponseTokens} token response guard.`
      );
    }
    const badgeParts = [];
    if (retrievedMemories.length) {
      badgeParts.push(`${retrievedMemories.length} memories`);
    }
    if (reasoning) {
      badgeParts.push('reasoning');
    }
    if (truncated) {
      badgeParts.push('trimmed');
    }
    await appendMessage('assistant', response, {
      badge: badgeParts.length ? badgeParts.join(' • ') : undefined,
      metadata: { retrievedMemories, truncated }
    });
    updateProcessState('modelA', {
      status: 'Reply delivered',
      detail: `Last reply contained ~${estimateTokenCount(response)} tokens.`
    });
    setTimeout(() => updateModelConnectionStatus(), 1200);
  } catch (error) {
    console.error('Model error', error);
    const errorMessage = error.message || 'Unknown error occurred';
    addSystemMessage(`Model error: ${errorMessage}`);
    updateModelStatus('Model error', 'status-pill--idle');
    updateProcessState('modelA', { status: 'Error', detail: errorMessage });
    void recordLog('error', `Model A error: ${errorMessage}`, { level: 'error' });
    setTimeout(() => updateModelConnectionStatus(), 3000);
  } finally {
    modelAInFlight = false;
    setTimeout(() => {
      updateProcessState('modelA', { status: 'Idle', detail: 'Waiting for user prompt.' });
    }, 400);
  }
}

async function buildModelMessages(userEntry) {
  const messages = [];
  let retrievedMemories = [];
  const shouldAutoRetrieve = Boolean(config.autoInjectMemories);
  const reasoningActive = isReasoningModeActive();

  if (shouldAutoRetrieve) {
    const retrievalCap = reasoningActive
      ? Math.max(0, Math.min(3, config.retrievalCount || 3))
      : config.retrievalCount;
    if (retrievalCap !== 0) {
      retrievedMemories = await retrieveRelevantMemories(userEntry.content, retrievalCap);
    }
  }

  if (config.systemPrompt?.trim()) {
    messages.push({ role: 'system', content: config.systemPrompt.trim() });
  }

  if (shouldAutoRetrieve && retrievedMemories.length) {
    const compiled = retrievedMemories
      .map((item) => `${formatTimestamp(item.timestamp)} • ${item.role}: ${item.content}`)
      .join('\n');
    messages.push({
      role: 'system',
      content: `Relevant long-term memories:\n${compiled}`
    });
  }

  const recent = getRecentConversation({ reasoningMode: reasoningActive });
  for (const item of recent) {
    messages.push({ role: item.role, content: item.content });
  }

  messages.push({ role: 'user', content: userEntry.content });
  return { messages, retrievedMemories };
}

function getRecentConversation(options = {}) {
  if (!conversationLog.length) {
    return [];
  }
  const reasoningActive = isReasoningModeActive(options);
  const maxTurns = reasoningActive ? Math.min(6, conversationLog.length) : config.contextTurns; // CODEx: Phase II limit on reasoning context.
  const limit = Math.max(2, Math.min(maxTurns, conversationLog.length));
  return conversationLog.slice(-limit);
}

// CODEx: Strip heavyweight properties from metadata before persisting chat entries.
function sanitizeMetadata(metadata) {
  if (!metadata || typeof metadata !== 'object') {
    return undefined;
  }
  const sanitized = {};
  for (const [key, value] of Object.entries(metadata)) {
    if (value == null) {
      continue;
    }
    if (Array.isArray(value)) {
      if (key === 'retrievedMemories') {
        sanitized.retrievalCount = value.length;
        continue;
      }
      if (value.length > 12) {
        sanitized[`${key}Count`] = value.length;
        continue;
      }
      sanitized[key] = value.slice();
      continue;
    }
    sanitized[key] = value;
  }
  return Object.keys(sanitized).length ? sanitized : undefined;
}

async function appendMessage(role, content, options = {}) {
  const {
    badge,
    speak = role === 'assistant',
    metadata = {},
    persist = true,
    toFloating = true
  } = options;
  let timestamp = Date.now();
  while (conversationLog.some((item) => item.timestamp === timestamp)) {
    timestamp += 1;
  }

  const sanitizedMetadata = sanitizeMetadata(metadata) ?? {};
  const entry = {
    id: sanitizedMetadata.id ?? timestamp,
    role,
    content,
    timestamp,
    origin: sanitizedMetadata.origin ?? 'floating',
    mode: sanitizedMetadata.mode ?? MODE_CHAT,
    ...sanitizedMetadata
  };

  assignTurnNumber(entry, sanitizedMetadata.turnNumber);
  const normalizedId = normalizeMessageId(entry.id);
  if (sanitizedMetadata.pinned || pinnedMessageIds.has(normalizedId)) {
    entry.pinned = true;
    pinnedMessageIds.add(normalizedId);
  }

  conversationLog.push(entry);
  if (toFloating) {
    floatingMemory.push(entry);
    trimFloatingMemory();
  } else {
    updateMemoryStatus();
  }
  renderMessage(entry, { badge });

  if (persist) {
    await persistMessage(entry);
  }

  persistPinnedMessages();
  renderFloatingMemoryWorkbench();

  if (persist) {
    queueMemoryCheckpoint(MODE_CHAT, entry);
  }

  void persistChatRagSnapshot();

  if (role === 'assistant' && config.autoSpeak && speak) {
    void speakText(content);
  }

  return entry;
}

function renderMessage(entry, { badge } = {}) {
  const node = elements.messageTemplate.content.firstElementChild.cloneNode(true);
  const messageId = entry.id ?? entry.timestamp;
  node.dataset.messageId = String(messageId);
  node.querySelector('.message-role').textContent = entry.role.toUpperCase();
  const turnNode = node.querySelector('.message-turn');
  if (entry.turnNumber) {
    turnNode.textContent = `Turn ${entry.turnNumber}`;
  } else {
    turnNode.remove();
  }
  node.querySelector('.message-time').textContent = formatTimestamp(entry.timestamp);
  const badgeNode = node.querySelector('.message-badge');
  if (badge || (entry.origin && entry.origin !== 'floating') || entry.pinned) {
    badgeNode.textContent = badge || (entry.pinned ? 'pinned' : entry.origin);
  } else {
    badgeNode.remove();
  }
  node.querySelector('.message-content').textContent = entry.content;
  elements.chatWindow.appendChild(node);
  scrollContainerToBottom(elements.chatWindow);
}

function scrollContainerToBottom(container) {
  if (!container) return;
  const performScroll = () => {
    const scrollers = [];
    let node = container;
    while (node && node !== document.body) {
      if (node.dataset?.scrollContainer === 'true' || node === container) {
        scrollers.push(node);
      }
      node = node.parentElement;
    }
    const docScroller = document.scrollingElement || document.documentElement;
    if (docScroller) {
      scrollers.push(docScroller);
    }

    const lastChild = container.lastElementChild;
    for (const target of scrollers) {
      if (target === docScroller) {
        window.scrollTo({ top: docScroller.scrollHeight, behavior: 'smooth' });
      } else if (typeof target.scrollTo === 'function') {
        target.scrollTo({ top: target.scrollHeight, behavior: 'smooth' });
      } else {
        target.scrollTop = target.scrollHeight;
      }
    }
    if (lastChild && typeof lastChild.scrollIntoView === 'function') {
      lastChild.scrollIntoView({ block: 'end', behavior: 'smooth' });
    }
  };
  requestAnimationFrame(performScroll);
  window.setTimeout(performScroll, 180);
  window.setTimeout(performScroll, 360);
}

function formatTimestamp(timestamp) {
  return new Date(normalizeTimestamp(timestamp)).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function formatRelativeTime(timestamp) {
  const value = Number(timestamp);
  if (!Number.isFinite(value)) {
    return 'just now';
  }
  const deltaMs = Date.now() - value;
  if (!Number.isFinite(deltaMs) || deltaMs <= 4000) {
    return 'just now';
  }
  const seconds = Math.max(0, Math.round(deltaMs / 1000));
  if (seconds < 60) {
    return `${seconds}s ago`;
  }
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m ago`;
  }
  const hours = Math.round(minutes / 60);
  if (hours < 24) {
    return `${hours}h ago`;
  }
  const days = Math.round(hours / 24);
  return `${days}d ago`;
}

function normalizeTimestamp(value) {
  if (typeof value === 'number') return value;
  if (value instanceof Date) return value.getTime();
  const parsed = Number(new Date(value));
  if (Number.isNaN(parsed)) {
    console.warn(`normalizeTimestamp received invalid date value: ${value}, using current time`);
    return Date.now();
  }
  return parsed;
}

function approximateSize(entry) {
  return new TextEncoder().encode(`${entry.role}:${entry.content}`).length;
}

function calculateMemoryUsage() {
  return floatingMemory.reduce((total, entry) => total + approximateSize(entry), 0);
}

function trimFloatingMemory() {
  let total = calculateMemoryUsage();
  const maxBytes = config.memoryLimitMB * MB;
  let removed = 0;
  let pinnedBlock = false;
  while (total > maxBytes && floatingMemory.length) {
    const removalIndex = floatingMemory.findIndex((item) => !item.pinned);
    if (removalIndex === -1) {
      pinnedBlock = true;
      break;
    }
    const [shifted] = floatingMemory.splice(removalIndex, 1);
    moveEntryToArchive(shifted, { silent: true });
    removed += 1;
    total = calculateMemoryUsage();
  }
  if (removed > 0) {
    addSystemMessage(`Floating memory trimmed. ${removed} message(s) archived to long-term storage.`);
    persistPinnedMessages();
  } else if (pinnedBlock) {
    addSystemMessage('Floating memory is full but every item is pinned. Increase the budget or unpin items to free space.');
  }
  updateMemoryStatus(total);
  renderFloatingMemoryWorkbench();
}

function markMessageAsArchived(id) {
  const selectorId = escapeCssIdentifier(String(id));
  const article = elements.chatWindow.querySelector(`[data-message-id="${selectorId}"]`);
  if (article) {
    const badge = article.querySelector('.message-badge');
    if (badge) {
      badge.textContent = 'archived';
    } else {
      const badgeEl = document.createElement('span');
      badgeEl.className = 'message-badge';
      badgeEl.textContent = 'archived';
      article.querySelector('.message-meta').appendChild(badgeEl);
    }
  }
}

async function persistMessage(entry) {
  if (!db) return;
  const timestamp = normalizeTimestamp(entry.timestamp);
  entry.timestamp = timestamp;
  const payload = {
    id: entry.id ?? timestamp,
    role: entry.role,
    content: entry.content,
    timestamp,
    turnNumber: entry.turnNumber ?? null,
    pinned: Boolean(entry.pinned),
    origin: entry.origin ?? 'floating',
    mode: entry.mode ?? MODE_CHAT
  };
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['messages'], 'readwrite');
    const store = transaction.objectStore('messages');
    const request = store.put(payload);
    request.onsuccess = () => resolve();
    request.onerror = (event) => reject(event);
  });
}

async function putRagRecord(record) {
  if (!db) return;
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([RAG_STORE_NAME], 'readwrite');
    const store = transaction.objectStore(RAG_STORE_NAME);
    const request = store.put(record);
    request.onsuccess = () => resolve();
    request.onerror = (event) => {
      console.error('Failed to persist RAG record', event);
      reject(event);
    };
  });
}

// CODEx: Persist reflex diagnostics for later analysis.
async function putReflexHistory(record) {
  if (!db) return;
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([REFLEX_HISTORY_STORE], 'readwrite');
    const store = transaction.objectStore(REFLEX_HISTORY_STORE);
    const request = store.add(record);
    request.onsuccess = () => resolve();
    request.onerror = (event) => {
      console.error('Failed to persist reflex history', event);
      reject(event);
    };
  });
}

async function persistChatRagSnapshot() {
  if (!db) return;
  const sessionId = ensureRagSessionId(MODE_CHAT);
  if (!sessionId) return;
  const entries = conversationLog
    .slice(-MAX_CHAT_RAG_MESSAGES)
    .map((item) => ({
      id: item.id ?? item.timestamp,
      role: item.role,
      content: item.content,
      timestamp: normalizeTimestamp(item.timestamp),
      turnNumber: item.turnNumber ?? null,
      pinned: Boolean(item.pinned),
      origin: item.origin ?? 'floating'
    }));
  if (!entries.length) return;
  const record = {
    id: sessionId,
    mode: MODE_CHAT,
    type: 'snapshot',
    updatedAt: Date.now(),
    messages: entries
  };
  try {
    await putRagRecord(record);
    markRagSave(MODE_CHAT);
  } catch (error) {
    console.error('Failed to update chat RAG snapshot', error);
  }
}

async function persistDualRagSnapshot() {
  if (!db) return;
  const sessionId = ensureRagSessionId(MODE_ARENA);
  if (!sessionId) return;
  const entries = dualChatHistory
    .slice(-MAX_ARENA_RAG_MESSAGES)
    .map((item, index) => ({
      speaker: item.speaker,
      speakerName: getAgentDisplayName(item.speaker),
      turn: item.turnNumber ?? index + 1,
      content: item.content,
      timestamp: normalizeTimestamp(item.timestamp)
    }));
  const record = {
    id: sessionId,
    mode: MODE_ARENA,
    type: 'snapshot',
    updatedAt: Date.now(),
    seed: activeDualSeed,
    messages: entries
  };
  try {
    await putRagRecord(record);
    markRagSave(MODE_ARENA);
  } catch (error) {
    console.error('Failed to update arena RAG snapshot', error);
  }
}

function queueMemoryCheckpoint(mode, entry) {
  if (!entry || typeof entry.content !== 'string' || !entry.content.trim()) return;
  if (mode === MODE_CHAT && entry.role === 'system') return;
  const targetBuffer = mode === MODE_ARENA ? arenaCheckpointBuffer : chatCheckpointBuffer;
  targetBuffer.push({
    id: entry.id ?? entry.timestamp,
    role: entry.role ?? (mode === MODE_ARENA ? entry.speaker ?? 'arena' : 'memory'),
    speaker: entry.speaker ?? null,
    speakerName: entry.speakerName ?? null,
    content: entry.content,
    timestamp: normalizeTimestamp(entry.timestamp),
    turnNumber: entry.turnNumber ?? null,
    pinned: Boolean(entry.pinned)
  });
  while (targetBuffer.length >= RAG_CHECKPOINT_SIZE) {
    const chunk = targetBuffer.splice(0, RAG_CHECKPOINT_SIZE);
    void persistRagCheckpoint(mode, chunk);
  }
}

async function persistRagCheckpoint(mode, chunk) {
  if (!db || !Array.isArray(chunk) || !chunk.length) return;
  const sessionId = ensureRagSessionId(mode);
  if (!sessionId) return;
  const timestamp = Date.now();
  const payload = chunk.map((item, index) => {
    if (mode === MODE_ARENA) {
      return {
        id: item.id ?? `${timestamp}-${index}`,
        speaker: item.speaker ?? null,
        speakerName: item.speakerName ?? item.role ?? null,
        content: item.content,
        timestamp: normalizeTimestamp(item.timestamp),
        turn: item.turnNumber ?? index + 1,
        origin: 'checkpoint'
      };
    }
    return {
      id: item.id ?? `${timestamp}-${index}`,
      role: item.role ?? 'memory',
      content: item.content,
      timestamp: normalizeTimestamp(item.timestamp),
      turnNumber: item.turnNumber ?? null,
      pinned: Boolean(item.pinned),
      origin: 'checkpoint'
    };
  });
  const record = {
    id: `${sessionId}-checkpoint-${timestamp}`,
    mode,
    type: 'checkpoint',
    updatedAt: timestamp,
    messages: payload
  };
  try {
    await putRagRecord(record);
    markRagSave(mode);
  } catch (error) {
    console.error('Failed to persist RAG checkpoint', error);
  }
}

function flushCheckpointBuffer(mode) {
  const targetBuffer = mode === MODE_ARENA ? arenaCheckpointBuffer : chatCheckpointBuffer;
  if (!targetBuffer.length) return;
  const chunk = targetBuffer.splice(0, targetBuffer.length);
  void persistRagCheckpoint(mode, chunk);
}

function renderFloatingMemoryWorkbench() {
  if (!elements.floatingMemoryList) return;
  elements.floatingMemoryList.innerHTML = '';
  const entries = [...floatingMemory].sort((a, b) => normalizeTimestamp(a.timestamp) - normalizeTimestamp(b.timestamp));
  if (!entries.length) {
    const empty = document.createElement('p');
    empty.className = 'hint-text';
    empty.textContent = 'Floating buffer is empty.';
    elements.floatingMemoryList.append(empty);
    updateFloatingSummary();
    return;
  }
  for (const entry of entries) {
    elements.floatingMemoryList.append(createMemoryItem(entry));
  }
  updateFloatingSummary();
}

function updateFloatingSummary() {
  if (elements.floatingMemoryCount) {
    elements.floatingMemoryCount.textContent = String(floatingMemory.length);
  }
  if (elements.pinnedMemoryCount) {
    const pinnedCount = floatingMemory.filter((item) => item.pinned).length;
    elements.pinnedMemoryCount.textContent = String(pinnedCount);
  }
  if (elements.ragMemoryCount) {
    elements.ragMemoryCount.textContent = String(countRagMemories());
  }
  if (elements.ragSnapshotCount) {
    const snapshotCount = Number.isFinite(ragTelemetry.lastLoadRecords)
      ? ragTelemetry.lastLoadRecords
      : 0;
    elements.ragSnapshotCount.textContent = String(snapshotCount);
  }
  if (elements.ragFootprint) {
    elements.ragFootprint.textContent = formatMegabytes(calculateRagMemoryUsage());
  }
  if (elements.ragCompressionRatio) {
    const currentBytes = calculateMemoryUsage();
    const ragBytes = calculateRagMemoryUsage();
    const combinedBytes = currentBytes + ragBytes;
    const compression = combinedBytes > 0 ? 1 - currentBytes / combinedBytes : 0;
    elements.ragCompressionRatio.textContent = `${(Math.max(0, Math.min(1, compression)) * 100).toFixed(1)}%`;
  }
  updateRagStatusDisplay();
  updateRetrievalStats();
}

function isRagDerived(entry) {
  const origin = (entry?.origin ?? '').toString().toLowerCase();
  if (!origin) return false;
  return RAG_ORIGIN_TOKENS.some((token) => origin.includes(token));
}

function countRagMemories() {
  return floatingMemory.reduce((total, entry) => (isRagDerived(entry) ? total + 1 : total), 0);
}

function calculateRagMemoryUsage() {
  return floatingMemory.reduce((total, entry) => {
    if (!isRagDerived(entry)) return total;
    return total + approximateSize(entry);
  }, 0);
}

function updateRagStatusDisplay() {
  if (!elements.ragImportStatus) return;
  const { lastLoad, lastLoadCount, lastLoadRecords, lastLoadBytes, lastSave, lastSaveMode, staticFiles, staticBytes, loaded } =
    ragTelemetry;

  if (elements.ragStatusFlag) {
    if (loaded) {
      const fileNote =
        staticFiles && staticFiles > 0
          ? ` (${staticFiles} file${staticFiles === 1 ? '' : 's'})`
          : '';
      elements.ragStatusFlag.textContent = `RAG: Loaded${fileNote}`;
      elements.ragStatusFlag.className = 'status-pill status-pill--synced';
    } else {
      elements.ragStatusFlag.textContent = 'RAG: Not loaded';
      elements.ragStatusFlag.className = 'status-pill status-pill--alert';
    }
  }

  if (!lastLoad && !lastSave) {
    elements.ragImportStatus.textContent = config.autoInjectMemories
      ? 'RAG archives not loaded yet. Auto-injecting floating memories when they match.'
      : 'RAG archives not loaded yet. Auto-injection disabled; floating memory stays on standby.';
    return;
  }

  const segments = [];
  if (lastLoad) {
    const loadTime = new Date(lastLoad).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    if (lastLoadRecords > 0) {
      const memoryLabel = lastLoadCount === 1 ? 'memory' : 'memories';
      const recordLabel = lastLoadRecords === 1 ? 'snapshot' : 'snapshots';
      const loadDescriptor = lastLoadCount > 0 ? `${lastLoadCount} ${memoryLabel}` : 'existing memories';
      segments.push(`Loaded ${loadDescriptor} from ${lastLoadRecords} ${recordLabel} @ ${loadTime}.`);
    } else {
      segments.push(`Checked RAG archives @ ${loadTime}.`);
    }
  }

  if (lastSave) {
    const saveTime = new Date(lastSave).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const modeLabel = lastSaveMode === MODE_ARENA ? 'arena' : 'chat';
    segments.push(`Last autosave (${modeLabel}) ran at ${saveTime}.`);
  }

  const ragBytes = Number.isFinite(lastLoadBytes) && lastLoadBytes > 0 ? lastLoadBytes : calculateRagMemoryUsage();
  if (ragBytes > 0) {
    segments.push(`Current RAG footprint: ${formatMegabytes(ragBytes)} across ${countRagMemories()} memories.`);
  }

  if (staticFiles > 0) {
    segments.push(`Static files scanned: ${staticFiles} (${formatMegabytes(staticBytes || ragBytes)}).`);
  }

  if (ragTelemetry.staticErrors && ragTelemetry.staticErrors.length) {
    const errorLabel = ragTelemetry.staticErrors.length === 1 ? 'file' : 'files';
    segments.push(`Skipped ${ragTelemetry.staticErrors.length} ${errorLabel}; open the debug console for details.`);
  }

  if (!config.autoInjectMemories) {
    segments.push('Auto-injection disabled; floating memory stays on standby.');
  } else if (Number.isFinite(config.retrievalCount) && config.retrievalCount > 0) {
    segments.push(`Auto-injecting up to ${config.retrievalCount} memory snippet${config.retrievalCount === 1 ? '' : 's'} per prompt.`);
  } else {
    segments.push('Auto-injecting all matching memories per prompt.');
  }

  elements.ragImportStatus.textContent = segments.join(' ');
}

function updateChunkerStatus() {
  if (elements.unchunkedFileCount) {
    elements.unchunkedFileCount.textContent = String(chunkerState.unchunkedFiles.length);
  }
  if (elements.chunkedFileCount) {
    elements.chunkedFileCount.textContent = String(chunkerState.chunkedFiles.length);
  }
  if (elements.totalChunkCount) {
    elements.totalChunkCount.textContent = String(chunkerState.totalChunks);
  }
  if (elements.chunkerStatus) {
    if (chunkerState.isProcessing) {
      elements.chunkerStatus.textContent = '🔄 Processing files…';
      elements.chunkerStatus.style.color = '#ff6b35'; // Orange/red for processing
    } else if (chunkerState.unchunkedFiles.length > 0) {
      const pending = chunkerState.unchunkedFiles.length;
      const ramBacked = chunkerState.unchunkedFiles.filter((file) => file.source === 'ram').length;
      let status = `🟡 ${pending} file${pending === 1 ? '' : 's'} ready to chunk.`;
      if (ramBacked > 0) {
        status += ` ${ramBacked} cached in RAM disk.`;
      }
      elements.chunkerStatus.textContent = status;
      elements.chunkerStatus.style.color = '#ffd23f'; // Yellow for ready
    } else {
      const chunkedCount = chunkerState.chunkedFiles.length;
      const ramChunked = chunkerState.chunkedFiles.filter((file) => file.source === 'ram').length;
      if (chunkedCount > 0) {
        const ramNote = ramChunked > 0 ? ` (${ramChunked} cached in RAM disk)` : '';
        elements.chunkerStatus.textContent = `🟢 ${chunkedCount} chunked file${chunkedCount === 1 ? '' : 's'} ready${ramNote}.`;
      } else {
        elements.chunkerStatus.textContent = '🟢 No files to chunk. Scan the rag/ folder first.';
      }
      elements.chunkerStatus.style.color = '#06ffa5'; // Green for idle
    }
  }
}

function markRagSave(mode = MODE_CHAT) {
  ragTelemetry.lastSave = Date.now();
  ragTelemetry.lastSaveMode = mode;
  updateRagStatusDisplay();
}

function createMemoryItem(entry) {
  const node = document.createElement('article');
  node.className = 'memory-item';
  if (entry.pinned) {
    node.classList.add('memory-item--pinned');
  }
  const messageId = normalizeMessageId(entry.id ?? entry.timestamp);
  node.dataset.memoryId = messageId;

  const meta = document.createElement('div');
  meta.className = 'memory-item__meta';
  const title = document.createElement('strong');
  const turnLabel = entry.turnNumber ? `Turn ${entry.turnNumber}` : 'Turn —';
  title.textContent = `${entry.role.toUpperCase()} • ${turnLabel}`;
  const timestamp = document.createElement('span');
  timestamp.textContent = formatTimestamp(entry.timestamp);
  meta.append(title, timestamp);

  const preview = document.createElement('p');
  preview.className = 'memory-item__content';
  preview.textContent = formatMemoryPreview(entry.content);

  const actions = document.createElement('div');
  actions.className = 'memory-item__actions';
  const pinButton = document.createElement('button');
  pinButton.type = 'button';
  pinButton.dataset.action = entry.pinned ? 'unpin' : 'pin';
  pinButton.textContent = entry.pinned ? 'Unpin' : 'Pin';
  const archiveButton = document.createElement('button');
  archiveButton.type = 'button';
  archiveButton.dataset.action = 'evict';
  archiveButton.classList.add('danger');
  archiveButton.textContent = 'Archive';
  actions.append(pinButton, archiveButton);

  node.append(meta, preview, actions);
  return node;
}

function handleFloatingMemoryListClick(event) {
  const button = event.target.closest('button[data-action]');
  if (!button) return;
  const container = button.closest('[data-memory-id]');
  if (!container) return;
  const messageId = container.dataset.memoryId;
  if (!messageId) return;
  switch (button.dataset.action) {
    case 'pin':
      setPinnedState(messageId, true);
      break;
    case 'unpin':
      setPinnedState(messageId, false);
      break;
    case 'evict':
      archiveMemoryEntry(messageId);
      break;
    default:
      break;
  }
}

function setPinnedState(id, shouldPin) {
  const normalizedId = normalizeMessageId(id);
  if (shouldPin) {
    pinnedMessageIds.add(normalizedId);
  } else {
    pinnedMessageIds.delete(normalizedId);
  }

  const logEntry = conversationLog.find((item) => normalizeMessageId(item.id) === normalizedId);
  const floatEntry = floatingMemory.find((item) => normalizeMessageId(item.id) === normalizedId);
  if (logEntry) {
    logEntry.pinned = shouldPin;
  }
  if (floatEntry) {
    floatEntry.pinned = shouldPin;
  }

  persistPinnedMessages();
  updateMemoryStatus();
  renderFloatingMemoryWorkbench();

  const turnLabel = logEntry?.turnNumber || floatEntry?.turnNumber;
  const labelText = turnLabel ? `turn ${turnLabel}` : 'memory';
  addSystemMessage(`${shouldPin ? 'Pinned' : 'Unpinned'} ${labelText} in floating memory.`);
}

function archiveMemoryEntry(id, { silent = false } = {}) {
  const normalizedId = normalizeMessageId(id);
  const index = floatingMemory.findIndex((item) => normalizeMessageId(item.id) === normalizedId);
  if (index === -1) return false;
  const [entry] = floatingMemory.splice(index, 1);
  moveEntryToArchive(entry, { silent: true });
  updateMemoryStatus();
  renderFloatingMemoryWorkbench();
  persistPinnedMessages();
  if (!silent) {
    const turnLabel = entry?.turnNumber ? `turn ${entry.turnNumber}` : 'memory';
    addSystemMessage(`Archived ${turnLabel} from the floating buffer.`);
  }
  return true;
}

function archiveOldArenaTurns(fraction = WATCHDOG_ARCHIVE_PERCENT, { reason = 'watchdog', silent = false } = {}) {
  if (!Number.isFinite(fraction) || fraction <= 0) {
    return 0; // CODEx: Ignore invalid archive fractions.
  }
  const arenaEntries = floatingMemory
    .filter((entry) => entry && entry.origin === 'arena' && !entry.pinned)
    .sort((a, b) => normalizeTimestamp(a.timestamp) - normalizeTimestamp(b.timestamp)); // CODEx: Oldest-first ordering.
  if (!arenaEntries.length) {
    return 0; // CODEx: Nothing to archive.
  }
  const removeCount = Math.max(1, Math.floor(arenaEntries.length * fraction)); // CODEx: Ensure at least one record.
  let archived = 0;
  for (const entry of arenaEntries.slice(0, removeCount)) {
    if (archiveMemoryEntry(entry.id, { silent: true })) {
      archived += 1; // CODEx: Track archived count for telemetry.
    }
  }
  if (archived > 0) {
    updateMemoryStatus();
    renderFloatingMemoryWorkbench();
    persistPinnedMessages();
    void persistDualRagSnapshot();
    if (!silent) {
      const label = reason === 'reflex' ? 'Reflex' : 'Watchdog';
      addSystemMessage(
        `${label} archived ${archived} arena ${archived === 1 ? 'turn' : 'turns'} to rag-logs.`
      ); // CODEx: Surface auto-archive context.
    }
  }
  return archived;
}

function archiveUnpinnedFloatingMemory() {
  const removable = floatingMemory.filter((item) => !item.pinned);
  if (!removable.length) {
    addSystemMessage('No unpinned memories to archive.');
    return;
  }
  let removed = 0;
  for (const entry of [...removable]) {
    if (archiveMemoryEntry(entry.id, { silent: true })) {
      removed += 1;
    }
  }
  updateMemoryStatus();
  renderFloatingMemoryWorkbench();
  persistPinnedMessages();
  addSystemMessage(`Manually archived ${removed} unpinned ${removed === 1 ? 'memory' : 'memories'}.`);
}

function handleStandaloneWatchdogMessage(event) {
  if (!event || typeof event.data !== 'object') {
    return;
  }
  const data = event.data;
  if (!data || data.source !== WATCHDOG_MESSAGE_SOURCE || data.type !== WATCHDOG_MESSAGE_TYPE) {
    return;
  }
  runStandaloneMemoryWatchdog();
}

function runStandaloneMemoryWatchdog() {
  if (typeof window === 'undefined') return;
  if (!window.standaloneStore && !window.standaloneFs) return;
  if (!isDualChatRunning) return;
  const limitBytes = config.memoryLimitMB * MB;
  if (!Number.isFinite(limitBytes) || limitBytes <= 0) return;
  const totalBytes = calculateMemoryUsage();
  if (totalBytes <= 0) return;
  const utilization = totalBytes / limitBytes;
  if (!Number.isFinite(utilization) || utilization < WATCHDOG_PRESSURE_THRESHOLD) {
    return;
  }
  if (lastWatchdogArchiveAt && Date.now() - lastWatchdogArchiveAt < 60000) {
    return;
  }
  const archived = archiveOldArenaTurns(WATCHDOG_ARCHIVE_PERCENT, { reason: 'watchdog', silent: true });
  if (archived === 0) {
    return;
  }
  lastWatchdogArchiveAt = Date.now();
  const percent = (utilization * 100).toFixed(1);
  addSystemMessage(
    `Watchdog archived ${archived} arena ${archived === 1 ? 'turn' : 'turns'} after floating memory hit ${percent}% of its budget.`
  );
  void recordLog('arena', `Watchdog archived ${archived} arena turn(s) at ${percent}% utilization.`, { silent: true });
}

function moveEntryToArchive(entry, { silent = false } = {}) {
  if (!entry) return false;
  const normalizedId = normalizeMessageId(entry.id ?? entry.timestamp);
  entry.origin = 'archived';
  entry.pinned = false;
  const logEntry = conversationLog.find((item) => normalizeMessageId(item.id) === normalizedId);
  if (logEntry && logEntry !== entry) {
    logEntry.origin = 'archived';
    logEntry.pinned = false;
  }
  pinnedMessageIds.delete(normalizedId);
  markMessageAsArchived(normalizedId);
  if (!silent) {
    persistPinnedMessages();
    updateMemoryStatus();
    renderFloatingMemoryWorkbench();
  }
  return true;
}

function persistPinnedMessages() {
  try {
    storageSetItem(PINNED_STORAGE_KEY, JSON.stringify(Array.from(pinnedMessageIds)));
  } catch (error) {
    console.error('Failed to persist pinned messages:', error);
  }
}

function normalizeMessageId(id) {
  if (typeof id === 'string') return id;
  if (typeof id === 'number') return String(id);
  if (id instanceof Date) return String(id.getTime());
  return String(id ?? '');
}

function escapeCssIdentifier(value) {
  if (window.CSS && typeof window.CSS.escape === 'function') {
    return window.CSS.escape(value);
  }
  return value.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}

function assignTurnNumber(entry, providedTurn) {
  const numeric = Number.parseInt(providedTurn, 10);
  if (!Number.isNaN(numeric) && numeric > 0) {
    entry.turnNumber = numeric;
    if (numeric > turnCounter) {
      turnCounter = numeric;
    }
    return;
  }

  if (typeof entry.turnNumber === 'number') {
    if (entry.turnNumber > turnCounter) {
      turnCounter = entry.turnNumber;
    }
    return;
  }

  if (entry.role === 'user') {
    turnCounter += 1;
    entry.turnNumber = turnCounter;
  } else if (entry.role === 'assistant') {
    if (turnCounter === 0) {
      turnCounter = 1;
    }
    entry.turnNumber = turnCounter;
  }
}

function formatMemoryPreview(text, limit = 220) {
  const content = (text ?? '').trim();
  if (content.length <= limit) return content;
  return `${content.slice(0, limit - 1)}…`;
}

function formatMegabytes(bytes, decimals = 2) {
  if (!Number.isFinite(bytes)) return '0 MB';
  const value = bytes / MB;
  const fixed = value >= 10 ? value.toFixed(Math.max(1, Math.min(2, decimals))) : value.toFixed(decimals);
  return `${fixed} MB`;
}

function updateMemoryStatus(currentBytes = calculateMemoryUsage()) {
  const used = (currentBytes / MB).toFixed(1);
  const totalCount = floatingMemory.length;
  const pinnedCount = floatingMemory.filter((item) => item.pinned).length;
  const messageLabel = totalCount === 1 ? 'msg' : 'msgs';
  const ragCount = countRagMemories();
  const ragLabel = ragCount ? ` • ${ragCount} from RAG` : '';
  const ragBytes = calculateRagMemoryUsage();
  const combinedBytes = currentBytes + ragBytes;
  const compression = combinedBytes > 0 ? 1 - currentBytes / combinedBytes : 0;
  const compressionPercent = Math.max(0, Math.min(1, compression));
  const compressionLabel = `${(compressionPercent * 100).toFixed(1)}% compressed`;
  elements.memoryStatus.innerHTML = `Floating memory: ${used} / ${config.memoryLimitMB}&nbsp;MB • ${totalCount} ${messageLabel} (${pinnedCount} pinned)${ragLabel} • ${compressionLabel}`;
  if (elements.ragCompressionRatio) {
    elements.ragCompressionRatio.textContent = `${(compressionPercent * 100).toFixed(1)}%`;
  }
}

async function runDiagnostics() {
  const startedAt = new Date();
  const preset = providerPresetMap.get(config.providerPreset) ?? providerPresetMap.get('custom');
  const ttsPreset = ttsPresetMap.get(config.ttsPreset) ?? ttsPresetMap.get(defaultConfig.ttsPreset);
  const floatingBytes = calculateMemoryUsage();
  const limitBytes = config.memoryLimitMB * MB;
  const percentUsed = limitBytes > 0 ? Math.min(100, (floatingBytes / limitBytes) * 100) : 0;
  const pinnedCount = floatingMemory.filter((item) => item.pinned).length;
  const archivedCount = Math.max(0, conversationLog.length - floatingMemory.length);
  const limitDisplay = hasDualTurnLimit() ? String(config.dualTurnLimit) : '∞';
  const arenaStatus = isDualChatRunning
    ? `running (${dualTurnsCompleted}/${limitDisplay} turns)`
    : `idle (${limitDisplay} limit)`;
  const agentAConnection = getAgentConnection('A');
  const agentBConnection = getAgentConnection('B');

  const diagnostics = [
    `• Provider preset: ${preset?.label ?? 'Custom'}`,
    `• Endpoint: ${config.endpoint || 'Not set'}`,
    `• Model: ${config.model || 'Not set'}`,
    `• API key: ${preset?.requiresKey ? (config.apiKey ? 'present' : 'missing') : config.apiKey ? 'set (optional)' : 'not required'}`,
    `• Max response tokens: ${Number.isFinite(config.maxResponseTokens) && config.maxResponseTokens > 0 ? config.maxResponseTokens : 'provider default'}`,
    `• Floating memory: ${floatingMemory.length} message(s), ${formatMegabytes(floatingBytes)} used (${percentUsed.toFixed(1)}% of ${config.memoryLimitMB} MB, ${pinnedCount} pinned)`,
    `• Archived log: ${archivedCount} message(s) in IndexedDB • ${conversationLog.length} total`,
    `• Speech preset: ${ttsPreset?.label ?? 'Browser voice'}${config.ttsServerUrl ? ` @ ${config.ttsServerUrl}` : ''}`,
    `• Auto speech: ${config.autoSpeak ? 'enabled' : 'disabled'} at ${config.ttsVolume}% volume`,
    `• Dual-agent arena: ${arenaStatus}`,
    `• Arena ${formatArenaConnectionDiagnostic('A', agentAConnection)}`,
    `• Arena ${formatArenaConnectionDiagnostic('B', agentBConnection)}`
  ];
  const retrievalDescriptor = !config.autoInjectMemories
    ? 'manual (promote memories when needed)'
    : Number.isFinite(config.retrievalCount) && config.retrievalCount > 0
      ? `auto (up to ${config.retrievalCount} snippet${config.retrievalCount === 1 ? '' : 's'})`
      : 'auto (all matching memories)';
  diagnostics.push(`• Memory retrieval: ${retrievalDescriptor}`);

  const callCount = modelCallMetrics.count;
  const averageMs = callCount > 0 ? Math.round(modelCallMetrics.totalMs / callCount) : 0;
  const timeoutLabel = `${modelCallMetrics.reasoningTimeouts} reasoning timeout${modelCallMetrics.reasoningTimeouts === 1 ? '' : 's'}`;
  diagnostics.push(
    callCount > 0
      ? `• Model calls: ${callCount} run${callCount === 1 ? '' : 's'} averaging ${averageMs} ms (${timeoutLabel})`
      : `• Model calls: no runs recorded yet (${timeoutLabel})`
  );

  const ragParts = [];
  if (Number.isFinite(ragTelemetry.lastRetrievalMs) && ragTelemetry.lastRetrievalMs > 0) {
    ragParts.push(`${ragTelemetry.lastRetrievalMs} ms last retrieval`);
  }
  if (ragTelemetry.lastLoadRecords) {
    ragParts.push(`${ragTelemetry.lastLoadRecords} chunk${ragTelemetry.lastLoadRecords === 1 ? '' : 's'} loaded`);
  }
  if (ragTelemetry.lastLoadBytes) {
    ragParts.push(`${formatMegabytes(ragTelemetry.lastLoadBytes)} footprint`);
  }
  if (ragTelemetry.lastLoad) {
    ragParts.push(`updated ${formatTimestamp(ragTelemetry.lastLoad)}`);
  }
  diagnostics.push(`• RAG telemetry: ${ragParts.length ? ragParts.join(' • ') : 'no retrieval activity yet'}`);

  const reflexLabel = config.reflexEnabled
    ? `enabled every ${config.reflexInterval} turn(s)`
    : 'disabled';
  diagnostics.push(`• Reflex mode: ${reflexLabel}`);
  const entropyDetails = describeEntropy(currentEntropyScore);
  const volleyLabel = `volley${config.entropyWindow === 1 ? '' : 's'}`;
  diagnostics.push(
    `• Entropy (last ${config.entropyWindow} ${volleyLabel}): ${(currentEntropyScore * 100).toFixed(0)}% ${entropyDetails.label}`
  );

  const endpointResult = await probeModelEndpoint();
  const statusEmoji = endpointResult.status === 'ok' ? '✅' : endpointResult.status === 'warning' ? '⚠️' : endpointResult.status === 'error' ? '❌' : 'ℹ️';
  diagnostics.push(`• Endpoint probe: ${statusEmoji} ${endpointResult.message}`);

  const agentProbeEntries = [];
  const agentAProbe = agentAConnection.inherits
    ? { status: 'info', message: 'Sharing main connection.' }
    : await probeModelEndpoint(agentAConnection);
  agentProbeEntries.push({ agent: 'A', result: agentAProbe });

  const agentBProbe = agentBConnection.inherits
    ? { status: 'info', message: 'Sharing main connection.' }
    : await probeModelEndpoint(agentBConnection);
  agentProbeEntries.push({ agent: 'B', result: agentBProbe });

  for (const entry of agentProbeEntries) {
    const label = getAgentDisplayName(entry.agent);
    const emoji = entry.result.status === 'ok' ? '✅' : entry.result.status === 'warning' ? '⚠️' : entry.result.status === 'error' ? '❌' : 'ℹ️';
    diagnostics.push(`• ${label} probe: ${emoji} ${entry.result.message}`);
  }

  diagnostics.push(`• Request timeout: ${config.requestTimeoutSeconds}s (reasoning: ${config.reasoningTimeoutSeconds}s)`);

  const diagnosticPayload = {
    timestamp: startedAt.toISOString(),
    model: {
      preset: preset?.id ?? 'custom',
      endpoint: config.endpoint,
      model: config.model,
      callCount,
      averageMs,
      totalMs: modelCallMetrics.totalMs,
      reasoningTimeouts: modelCallMetrics.reasoningTimeouts
    },
    rag: {
      lastRetrievalMs: ragTelemetry.lastRetrievalMs,
      lastLoadCount: ragTelemetry.lastLoadCount,
      lastLoadRecords: ragTelemetry.lastLoadRecords,
      lastLoadBytes: ragTelemetry.lastLoadBytes,
      lastLoad: ragTelemetry.lastLoad
    },
    memory: {
      floatingBytes,
      limitBytes,
      pinnedCount,
      floatingCount: floatingMemory.length,
      ragCount: countRagMemories()
    },
    arena: {
      running: isDualChatRunning,
      turnsCompleted: dualTurnsCompleted,
      entropy: currentEntropyScore
    },
    probes: {
      main: endpointResult,
      agents: agentProbeEntries
    }
  };

  const report = `Diagnostics @ ${startedAt.toLocaleTimeString()}\n${diagnostics.join('\n')}`;
  addSystemMessage(report);

  console.info('[Diagnostics]', JSON.stringify(diagnosticPayload, null, 2));

  if (endpointResult.status === 'ok') {
    updateModelConnectionStatus();
  } else if (endpointResult.status === 'error') {
    updateModelStatus(`Model check failed • ${endpointResult.message}`, 'status-pill--idle');
  }
}

async function probeModelEndpoint(overrides = {}) {
  const endpoint = overrides.endpoint ?? config.endpoint;
  const apiKey = overrides.apiKey ?? config.apiKey;
  if (!endpoint) {
    return { status: 'skipped', message: 'No endpoint configured.' };
  }
  const headers = {};
  if (apiKey) {
    headers.Authorization = `Bearer ${apiKey.trim()}`;
  }
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 4000);
  try {
    const response = await fetch(endpoint, {
      method: 'OPTIONS',
      headers,
      signal: controller.signal,
      mode: 'cors'
    });
    clearTimeout(timeoutId);
    if (response.ok) {
      const allow = response.headers.get('allow') || response.headers.get('Access-Control-Allow-Methods');
      const allowText = allow ? ` (allows: ${allow})` : '';
      return { status: 'ok', message: `Reachable • HTTP ${response.status}${allowText}` };
    }
    return { status: 'warning', message: `Endpoint responded with HTTP ${response.status}` };
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      return { status: 'warning', message: 'Timed out after 4s while probing endpoint.' };
    }
    return { status: 'error', message: error?.message || 'Probe failed.' };
  }
}

function addSystemMessage(text) {
  void appendMessage('system', text, {
    badge: 'system',
    persist: false,
    speak: false,
    toFloating: false,
    metadata: { origin: 'system' }
  });
}

async function promoteRelevantMemories() {
  const query = prompt('Describe what you want to recall from long-term memory:');
  if (!query) return;
  const memories = await retrieveRelevantMemories(query, config.retrievalCount);
  if (!memories.length) {
    addSystemMessage('No relevant memories found for that query.');
    return;
  }
  for (const memory of memories) {
    const normalizedId = normalizeMessageId(memory.id ?? memory.timestamp);
    if (floatingMemory.some((item) => normalizeMessageId(item.id) === normalizedId)) {
      continue;
    }
    const linked = conversationLog.find((item) => normalizeMessageId(item.id) === normalizedId);
    const target = linked ?? { ...memory };
    target.origin = memory.origin ?? 'promoted';
    if (pinnedMessageIds.has(normalizedId)) {
      target.pinned = true;
    }
    floatingMemory.push(target);
  }
  trimFloatingMemory();
  addSystemMessage(`Promoted ${memories.length} memory snippet(s) into the floating context.`);
}

async function retrieveRelevantMemories(query, limit, options = {}) {
  const normalizedQuery = (query ?? '').trim();
  const tokens = tokenize(normalizedQuery);
  if (!tokens.length) return [];
  const retrievalStarted = getTimestampMs();
  const channel = resolveEmbeddingChannel({ speaker: options.speaker, bucketKey: options.bucketKey }); // CODEx: Determine retrieval channel for embeddings.

  const candidates = new Map();
  for (const entry of floatingMemory) {
    const id = normalizeMessageId(entry.id ?? entry.timestamp);
    candidates.set(id, entry);
  }

  const stored = await getAllMessages();
  for (const entry of stored) {
    const normalizedTimestamp = normalizeTimestamp(entry.timestamp);
    const id = normalizeMessageId(entry.id ?? normalizedTimestamp);
    if (!candidates.has(id)) {
      const logMatch = conversationLog.find((item) => normalizeMessageId(item.id) === id);
      if (logMatch) {
        candidates.set(id, logMatch);
      } else {
        candidates.set(id, {
          id,
          role: entry.role,
          content: entry.content,
          timestamp: normalizedTimestamp,
          origin: 'long-term',
          turnNumber: entry.turnNumber ?? undefined,
          pinned: pinnedMessageIds.has(id)
        });
      }
    }
  }
  const candidateList = Array.from(candidates.values()); // CODEx: Normalize candidate map into an iterable list.
  let useEmbeddings = isEmbeddingRetrievalEnabled() && embeddingServiceHealthy; // CODEx: Determine if vector search should run.
  let queryVector = null; // CODEx: Placeholder for query embedding when available.
  if (useEmbeddings) { // CODEx: Only compute embeddings when enabled and healthy.
    queryVector = await generateQueryEmbedding(normalizedQuery, tokens, { bucketKey: channel.bucket, speaker: channel.speaker }); // CODEx: Derive query embedding aligned with the active channel.
    if (!Array.isArray(queryVector) || !queryVector.length || !embeddingServiceHealthy) { // CODEx: Disable embeddings when unavailable.
      useEmbeddings = false; // CODEx: Fall back to lexical scoring if embeddings failed.
    } // CODEx
  } // CODEx

  const scored = await vectorSearch({ // CODEx: Blend vector and lexical scoring through shared helper.
    candidates: candidateList, // CODEx: Supply full candidate set for scoring.
    limit, // CODEx: Honor retrieval limit requested by caller.
    useEmbeddings, // CODEx: Toggle embedding math based on availability.
    queryVector, // CODEx: Provide precomputed query embedding when available.
    ensureEmbedding: (entry) => ensureEntryEmbedding(entry, { bucketKey: channel.bucket }), // CODEx: Resolve candidate embeddings within the channel bucket.
    lexicalScorer: (entry) => cosineSimilarity(tokens, tokenize(entry.content || '')), // CODEx: Compute lexical overlap per entry with null safety.
    similarity: cosineSimilarity, // CODEx: Apply unified cosine similarity for vector comparison.
    debugHook: (error, entry) => console.debug('Embedding match fallback', { error, id: entry.id }) // CODEx: Surface embedding failures for debugging.
  });

  const selected = limit > 0 ? scored.slice(0, limit) : scored; // CODEx: Respect caller-imposed top-k limit.
  const result = selected.map((item) => item.entry); // CODEx: Extract memory payloads for downstream use.
  ragTelemetry.lastRetrievalMs = Math.round(Math.max(0, getTimestampMs() - retrievalStarted)); // CODEx: Record retrieval latency for diagnostics.
  ragTelemetry.lastRetrievalMode = useEmbeddings ? 'embedding' : 'lexical'; // CODEx: Persist retrieval mode for telemetry overlays.

  if (typeof console !== 'undefined' && typeof console.debug === 'function') { // CODEx: Avoid logging when console missing.
    const topResults = selected.slice(0, Math.min(5, selected.length)).map((item) => ({ // CODEx: Summarize top matches for debugging output.
      id: item.entry.id ?? item.entry.timestamp, // CODEx: Provide stable identifier for the retrieved memory.
      score: Number.isFinite(item.score) ? Number(item.score).toFixed(3) : '0.000', // CODEx: Present combined score with consistent precision.
      origin: item.entry.origin ?? 'unknown' // CODEx: Include memory origin to aid troubleshooting.
    }));
    console.debug('RAG retrieval', { // CODEx: Emit retrieval diagnostics to the debug console.
      query: normalizedQuery.slice(0, 160), // CODEx: Log truncated query context for reference.
      latencyMs: ragTelemetry.lastRetrievalMs, // CODEx: Share measured retrieval latency.
      mode: ragTelemetry.lastRetrievalMode, // CODEx: Indicate whether embeddings or lexical fallback was used.
      topResults // CODEx: Provide the scored top-k results for debugging.
    });
  }

  return result; // CODEx: Return ordered memory results to callers.
}

function tokenize(text) {
  return (text || '')
    .toLowerCase()
    .match(/[a-z0-9]+/g)
    ?.filter((token) => token.length > 1) || [];
}

// CODEx: Unified cosine similarity hot-fix supporting numeric vectors and lexical token sets.
function cosineSimilarity(a, b) {
  if (Array.isArray(a) && typeof a[0] === 'number') {
    const dot = a.reduce((sum, value, index) => sum + value * (b?.[index] ?? 0), 0);
    const magA = Math.sqrt(a.reduce((sum, value) => sum + value * value, 0));
    const magB = Math.sqrt((Array.isArray(b) ? b : []).reduce((sum, value) => sum + value * value, 0));
    if (!magA || !magB) {
      return 0;
    }
    return dot / (magA * magB);
  }
  const setA = new Set(Array.isArray(a) ? a : []);
  const setB = new Set(Array.isArray(b) ? b : []);
  if (!setA.size || !setB.size) {
    return 0;
  }
  const overlap = [...setA].filter((token) => setB.has(token)).length;
  return overlap / Math.sqrt(setA.size * setB.size);
}

// CODEx: Generate or reuse embeddings for the active query text.
async function generateQueryEmbedding(query, tokens, options = {}) {
  if (!isEmbeddingRetrievalEnabled()) {
    return null; // CODEx: Skip embedding generation when disabled by configuration.
  }
  if (!query || query.length < EMBEDDING_MIN_QUERY_LENGTH) {
    return null; // CODEx: Avoid embedding tiny prompts to conserve provider usage.
  }
  try {
    return await generateTextEmbedding(query, options); // CODEx: Delegate to channel-aware embedding resolver.
  } catch (error) {
    console.debug('Embedding query fallback engaged:', error); // CODEx: Surface provider failures for diagnostics.
    return null; // CODEx: Allow lexical scoring path to handle retrieval when embeddings fail.
  }
}

// CODEx: Ensure a message has an embedding cached for future retrieval.
async function ensureEntryEmbedding(entry, options = {}) {
  const bucketKey = normalizeEmbeddingBucket(options.bucketKey); // CODEx: Normalize bucket selection for retrieval context.
  const baseId = normalizeMessageId(entry.id ?? entry.timestamp ?? entry.content); // CODEx: Stable identifier for caching.
  const cacheKey = `${bucketKey}::${baseId}`; // CODEx: Namespaced cache key per bucket to avoid cross-talk.
  if (messageEmbeddingCache.has(cacheKey)) { // CODEx: Reuse memoized embedding when present.
    return messageEmbeddingCache.get(cacheKey); // CODEx: Return cached embedding vector.
  }
  if (entry?.metadata?.embeddingKey) { // CODEx: Attempt to reuse chunk embeddings by key.
    const chunkKey = entry.metadata.embeddingKey; // CODEx: Extract chunk cache identifier.
    const cachedVector = resolveCachedEmbedding(bucketKey, chunkKey); // CODEx: Merge RAM and disk caches per bucket.
    if (Array.isArray(cachedVector) && cachedVector.length) { // CODEx: Validate recovered vector.
      messageEmbeddingCache.set(cacheKey, cachedVector); // CODEx: Cache resolved vector for future lookups.
      return cachedVector; // CODEx: Return cached embedding without recomputation.
    }
  }
  const content = (entry?.content ?? '').trim();
  if (!content) {
    return null;
  }
  try {
    const vector = await generateTextEmbedding(content, { bucketKey }); // CODEx: Generate embeddings scoped to the active bucket.
    if (vector) {
      messageEmbeddingCache.set(cacheKey, vector); // CODEx: Memoize under bucket-qualified key.
      return vector;
    }
  } catch (error) {
    console.debug('Message embedding fallback:', error);
  }
  const fallbackVector = lexicalEmbedding(content);
  messageEmbeddingCache.set(cacheKey, fallbackVector); // CODEx: Cache lexical vector for future lookups.
  return fallbackVector;
}

// CODEx: Produce or reuse an embedding for arbitrary text content.
async function generateTextEmbedding(text, options = {}) {
  const trimmed = (text ?? '').trim();
  if (!trimmed) {
    return null;
  }
  const channel = resolveEmbeddingChannel(options); // CODEx: Determine provider, model, and bucket for this embedding request.
  const cacheKey = `${channel.bucket}::${channel.model}::${hashString(trimmed)}`; // CODEx: Cache key scoped by bucket and model.
  if (embeddingCache.has(cacheKey)) { // CODEx: Reuse cached embedding when available.
    return embeddingCache.get(cacheKey);
  }
  let vector = null;
  if (!isEmbeddingRetrievalEnabled()) {
    vector = lexicalEmbedding(trimmed); // CODEx: Produce lexical vector immediately when embeddings disabled.
  } else {
    try {
      vector = await requestEmbeddingFromProvider(trimmed, channel.model, {
        signal: options.signal,
        providerOverride: channel.useDetection ? null : channel.provider,
        baseUrlOverride: channel.baseUrl || options.baseUrlOverride,
        endpointOverride: channel.endpoint || options.endpointOverride,
        apiKeyOverride: channel.apiKey || options.apiKeyOverride,
        bucketKey: channel.bucket,
        contextLength: channel.contextLength
      }); // CODEx: Request embedding via the resolved channel.
      if (Array.isArray(vector) && vector.length) {
        setEmbeddingServiceHealth(true); // CODEx: Mark embedding service healthy on success.
      } else {
        setEmbeddingServiceHealth(false, 'empty embedding'); // CODEx: Flag provider failure when response lacks vector data.
      }
    } catch (error) {
      console.debug('Embedding provider unavailable, falling back to lexical vector:', error); // CODEx: Log provider outage for debugging.
      setEmbeddingServiceHealth(false, error?.message || 'offline'); // CODEx: Persist failure reason for UI indicator.
    }
    if (!Array.isArray(vector) || !vector.length) {
      vector = lexicalEmbedding(trimmed); // CODEx: Fallback to lexical vector when provider fails.
    }
  }
  embeddingCache.set(cacheKey, vector);
  return vector;
}

// CODEx: Resolve an embedding endpoint that matches the active provider.
function resolveEmbeddingEndpoint() {
  const configured = (config.embeddingEndpoint ?? '').trim();
  if (configured) {
    return configured;
  }
  const endpoint = (config.endpoint ?? '').trim();
  if (!endpoint) {
    return '';
  }
  if (isOllamaCloud(endpoint)) { // CODEx: Map Ollama Cloud chat endpoints to embeddings path.
    if (/\/v1\/chat\/completions$/i.test(endpoint)) { // CODEx: Handle OpenAI-compatible pathing.
      return endpoint.replace(/\/v1\/chat\/completions$/i, '/v1/embeddings'); // CODEx: Swap chat completions with embeddings endpoint.
    }
    if (/\/v1\/chat$/i.test(endpoint)) { // CODEx: Support condensed chat route.
      return endpoint.replace(/\/v1\/chat$/i, '/v1/embeddings'); // CODEx: Normalize to embeddings endpoint.
    }
    return `${stripTrailingSlashes(endpoint)}/v1/embeddings`; // CODEx: Default cloud embeddings suffix.
  }
  if (isOllamaLocal(endpoint)) { // CODEx: Map Ollama local chat endpoints to embeddings path.
    if (/\/v1\/chat\/completions$/i.test(endpoint)) { // CODEx: Handle OpenAI-style local endpoints.
      return endpoint.replace(/\/v1\/chat\/completions$/i, '/api/embeddings'); // CODEx: Translate to native /api/embeddings path.
    }
    if (/\/api\/chat$/i.test(endpoint)) { // CODEx: Handle /api/chat local endpoints.
      return endpoint.replace(/\/api\/chat$/i, '/api/embeddings'); // CODEx: Swap chat route for embeddings route.
    }
    return `${stripTrailingSlashes(endpoint)}/api/embeddings`; // CODEx: Append embeddings suffix for custom bases.
  }
  if (isLikelyLmStudio(endpoint)) {
    return endpoint.replace(/\/v1\/chat\/completions/, '/api/embeddings').replace(/\/api\/chat$/, '/api/embeddings');
  }
  if (/\/chat\/completions/.test(endpoint)) {
    return endpoint.replace(/\/chat\/completions/, '/embeddings');
  }
  if (endpoint.endsWith('/api/chat')) {
    return `${endpoint.slice(0, -5)}embeddings`;
  }
  return `${stripTrailingSlashes(endpoint)}/embeddings`;
}

// CODEx: Detect Ollama Cloud endpoints for schema adjustments.
function isOllamaCloud(endpoint) {
  return /api\.ollama\.ai/i.test(endpoint || '');
}

// CODEx: Detect Ollama local endpoints for schema adjustments.
function isOllamaLocal(endpoint) {
  if (!endpoint) {
    return false;
  }
  return /11434/.test(endpoint) || (/ollama/i.test(endpoint) && /localhost|127\.0\.0\.1/i.test(endpoint));
}

// CODEx: Detect Ollama endpoints for schema adjustments.
function isLikelyOllama(endpoint) {
  return isOllamaCloud(endpoint) || isOllamaLocal(endpoint);
}

// CODEx: Detect LM Studio endpoints for schema adjustments.
function isLikelyLmStudio(endpoint) {
  return /1234/.test(endpoint) || /lmstudio/i.test(endpoint);
}

// CODEx: Strip trailing slashes while preserving protocol prefixes.
function stripTrailingSlashes(value) {
  return value.replace(/\/+$/, '');
}

// CODEx: Reduce an endpoint to its base path.
function deriveBaseUrl(endpoint) {
  try {
    const parsed = new URL(endpoint, window.location.origin);
    const segments = parsed.pathname.split('/').filter(Boolean);
    if (segments.length > 1) {
      segments.pop();
    }
    parsed.pathname = segments.length ? `/${segments.join('/')}` : '/';
    parsed.search = '';
    parsed.hash = '';
    return parsed.toString().replace(/\/$/, '');
  } catch (error) {
    const withoutQuery = endpoint.split(/[?#]/)[0];
    return withoutQuery.replace(/\/[^/]*$/, '');
  }
}

// CODEx: Request embeddings from either local or cloud providers with graceful fallback.
async function requestEmbeddingFromProvider(text, model, options = {}) {
  const {
    signal,
    providerOverride = null,
    baseUrlOverride = '',
    endpointOverride = '',
    apiKeyOverride = '',
    contextLength
  } = options; // CODEx: Extend embedding requests with override metadata.
  const sharedApiKey = (config.embeddingApiKey || config.apiKey || '').trim(); // CODEx: Shared embedding key fallback.
  const manualApiKey = (apiKeyOverride || '').trim(); // CODEx: Manual override supplied by caller.
  const policyHeader = (config.openRouterPolicy ?? '').trim(); // CODEx: Data policy header for OpenRouter.
  const embeddingEndpointOverride = (config.embeddingEndpoint ?? '').trim(); // CODEx: Use configured embedding endpoint when provided.
  const remoteEndpoint = (() => { // CODEx: Derive remote fallback endpoint when applicable.
    if (embeddingEndpointOverride && !isLikelyLmStudio(embeddingEndpointOverride) && !isLikelyOllama(embeddingEndpointOverride)) {
      return embeddingEndpointOverride; // CODEx: Honor explicit remote override.
    }
    const baseEndpoint = (config.endpoint ?? '').trim(); // CODEx: Inspect primary model endpoint.
    if (baseEndpoint && !isLikelyLmStudio(baseEndpoint) && !isLikelyOllama(baseEndpoint)) {
      return resolveEmbeddingEndpoint(); // CODEx: Translate chat endpoint into embeddings path for remote providers.
    }
    return ''; // CODEx: Default to module-provided OpenAI endpoint.
  })();

  const manualBase = baseUrlOverride ? baseUrlOverride.trim() : ''; // CODEx: Normalize manual base override.
  const manualEndpoint = endpointOverride ? endpointOverride.trim() : ''; // CODEx: Normalize manual endpoint override.
  const fetchWithPolicy = (url, options = {}) => { // CODEx: Inject OpenRouter policy header when necessary.
    const next = { ...options, headers: { ...(options?.headers || {}) } }; // CODEx: Clone options to avoid mutation.
    if (policyHeader && /openrouter\.ai/.test(url)) { // CODEx: Match OpenRouter host heuristically.
      next.headers['X-OpenRouter-Data-Policy'] = policyHeader; // CODEx: Attach data policy header.
    }
    return fetch(url, next); // CODEx: Delegate to native fetch.
  };

  const overrideSnapshot = providerOverride
    ? { active: providerOverride, baseUrl: manualBase }
    : null; // CODEx: Snapshot for explicit provider overrides.
  const detectionSnapshot = overrideSnapshot ? null : await resolveActiveEmbeddingProvider(false); // CODEx: Probe environment only when overrides absent.
  const providerSnapshot = overrideSnapshot ?? detectionSnapshot ?? { active: EMBEDDING_PROVIDERS.OPENAI, baseUrl: '' }; // CODEx: Use detection or fall back to remote defaults.

  async function invokeProvider(provider, baseUrl, endpoint) { // CODEx: Shared helper for provider invocation.
    const appliesManual = !providerOverride || provider === providerOverride; // CODEx: Only apply manual overrides to matching providers.
    const resolvedBase = appliesManual && manualBase ? manualBase : (baseUrl || ''); // CODEx: Choose base URL override when applicable.
    const providerRouteKey = getProviderRouteKey(provider); // CODEx: Derive route mapping key for provider.
    const routeSuffix = EMBED_ROUTES[providerRouteKey] || EMBED_ROUTES['openai-like']; // CODEx: Resolve expected embedding route suffix.
    const defaultEndpoint = resolvedBase ? joinProviderEndpoint(resolvedBase, routeSuffix) : endpoint; // CODEx: Build default endpoint when base provided.
    const resolvedEndpoint = appliesManual && manualEndpoint ? manualEndpoint : (endpoint || defaultEndpoint); // CODEx: Choose endpoint override or default route.
    const resolvedKey = appliesManual && manualApiKey ? manualApiKey : sharedApiKey; // CODEx: Prefer manual API key when provided.
    console.log('[RAG] Provider route', { provider, endpoint: resolvedEndpoint || 'auto', baseUrl: resolvedBase || baseUrl || '', model }); // CODEx: Emit provider routing telemetry.
    return requestEmbeddingVector({
      provider,
      baseUrl: resolvedBase,
      endpointOverride: resolvedEndpoint,
      model,
      text,
      apiKey: resolvedKey,
      signal,
      logger: console,
      fetchImpl: fetchWithPolicy,
      contextLength
    });
  }

  try { // CODEx: Attempt primary provider request.
    const vector = await invokeProvider(
      providerSnapshot.active,
      providerSnapshot.baseUrl,
      overrideSnapshot ? '' : embeddingEndpointOverride
    );
    setEmbeddingServiceHealth(true); // CODEx: Mark provider healthy on success.
    return vector; // CODEx: Return resolved embedding vector.
  } catch (error) {
    console.warn('[RAG Provider] fallback triggered', { provider: providerSnapshot.active, error }); // CODEx: Log failure for diagnostics with provider context.
    setEmbeddingServiceHealth(false, error?.message || 'offline'); // CODEx: Update health indicator with failure reason.
    if (providerSnapshot.active !== EMBEDDING_PROVIDERS.OPENAI) { // CODEx: Only fallback when starting from local provider.
      try {
        const vector = await invokeProvider(
          EMBEDDING_PROVIDERS.OPENAI,
          '',
          remoteEndpoint
        );
        if (detectionSnapshot) { // CODEx: Promote remote provider only when detection provided snapshot.
          embeddingProviderState.active = EMBEDDING_PROVIDERS.OPENAI; // CODEx: Promote remote provider after fallback.
          embeddingProviderState.baseUrl = '';
          embeddingProviderState.lastProbe = Date.now();
        }
        setEmbeddingServiceHealth(true); // CODEx: Remote success restores healthy status.
        return vector;
      } catch (fallbackError) {
        console.warn('Remote embedding fallback failed', fallbackError); // CODEx: Surface remote failure.
        throw fallbackError; // CODEx: Propagate fallback failure to caller.
      }
    }
    throw error; // CODEx: Re-throw when primary provider is already remote.
  }
}

// CODEx: Produce hashed lexical embeddings as an offline fallback.
function lexicalEmbedding(input, dimensions = EMBEDDING_HASH_BUCKETS) {
  const tokens = Array.isArray(input) ? input : tokenize(input);
  if (!tokens.length) {
    return new Array(dimensions).fill(0);
  }
  const vector = new Array(dimensions).fill(0);
  for (const token of tokens) {
    const bucket = hashString(token) % dimensions;
    vector[bucket] += 1;
  }
  return normalizeVector(vector);
}

// CODEx: Derive lightweight topic fingerprints for Reflex storage.
function extractTopicFingerprints(text, limit = 5) {
  const tokens = tokenize(text)
    .filter((token) => token.length > 3)
    .map((token) => token.toLowerCase());
  if (!tokens.length) {
    return [];
  }
  const counts = new Map();
  for (const token of tokens) {
    counts.set(token, (counts.get(token) || 0) + 1);
  }
  const sorted = Array.from(counts.entries()).sort((a, b) => {
    if (b[1] === a[1]) {
      return a[0].localeCompare(b[0]);
    }
    return b[1] - a[1];
  });
  return sorted.slice(0, limit).map(([token]) => token);
}

// CODEx: Normalize vectors to unit length for cosine scoring.
function normalizeVector(vector) {
  let magnitude = 0;
  for (const value of vector) {
    if (Number.isFinite(value)) {
      magnitude += value * value;
    }
  }
  if (!magnitude) {
    return vector;
  }
  const scale = 1 / Math.sqrt(magnitude);
  return vector.map((value) => (Number.isFinite(value) ? value * scale : 0));
}

// CODEx: Simple 32-bit hash for caching and lexical buckets.
function hashString(value) {
  let hash = 0;
  const text = String(value);
  for (let index = 0; index < text.length; index += 1) {
    hash = (hash << 5) - hash + text.charCodeAt(index);
    hash |= 0;
  }
  return hash >>> 0;
}

// CODEx: Identify the connection strategy for the active endpoint.
function detectConnectionType(endpoint, preset) {
  const presetId = preset?.id ?? config.providerPreset;
  if (presetId === 'ollama-cloud' || isOllamaCloud(endpoint)) { // CODEx: Treat Ollama Cloud as OpenAI-compatible remote API.
    return 'openai';
  }
  if (presetId === 'ollama' || isOllamaLocal(endpoint)) {
    return 'ollama';
  }
  if (presetId === 'lmstudio' || isLikelyLmStudio(endpoint)) {
    return 'lmstudio';
  }
  return 'openai';
}

// CODEx: Derive the base local endpoint for LM Studio or Ollama.
function resolveLocalChatEndpoint(endpoint, type, modelName) {
  const base = stripTrailingSlashes(deriveBaseUrl(endpoint) || endpoint || ''); // CODEx: Normalize base endpoint for routing.
  const providerLabel = type === 'ollama' ? 'Ollama (Local)' : 'LM Studio'; // CODEx: Human-readable provider label for logs.
  const isEmbeddingModel = isEmbeddingModelId(modelName); // CODEx: Detect embedding-only checkpoints for guardrails.
  const url = (() => { // CODEx: Resolve provider-specific chat endpoint.
    if (type === 'ollama') {
      if (/\/api\/generate$/i.test(endpoint)) {
        return endpoint; // CODEx: Respect explicit generate endpoint overrides.
      }
      if (/\/v1\/chat\/completions$/i.test(endpoint)) {
        return endpoint.replace(/\/v1\/chat\/completions$/i, '/api/generate'); // CODEx: Translate legacy OpenAI path.
      }
      if (/\/api\/chat$/i.test(endpoint)) {
        return endpoint.replace(/\/api\/chat$/i, '/api/generate'); // CODEx: Normalize deprecated chat path.
      }
      return `${base}/api/generate`; // CODEx: Default Ollama local chat endpoint.
    }
    if (type === 'lmstudio') {
      if (/\/v1\/chat\/completions$/i.test(endpoint)) {
        return endpoint; // CODEx: Accept already-normalized LM Studio path.
      }
      if (/\/api\/chat$/i.test(endpoint)) {
        return endpoint.replace(/\/api\/chat$/i, '/v1/chat/completions'); // CODEx: Upgrade legacy API path.
      }
      return `${base}/v1/chat/completions`; // CODEx: Default LM Studio chat endpoint.
    }
    return endpoint; // CODEx: Fallback for unhandled provider types.
  })();
  if (isEmbeddingModel && /generate|chat\/completions/i.test(url)) {
    throw new Error(`Model ${modelName} does not support text generation`); // CODEx: Block embedding checkpoints from chat route.
  }
  console.log(`[Bridge] ${providerLabel} → ${url} :: ${modelName}`); // CODEx: Telemetry for provider routing decisions.
  return url;
}

// CODEx: Normalize chat messages for local APIs.
function normalizeLocalMessages(messages) {
  return messages.map((message) => ({ role: message.role, content: message.content }));
}

// CODEx: Convert a chat transcript into a single prompt for /api/generate fallbacks.
function collapseMessagesToPrompt(messages) {
  return messages.map((message) => `${message.role.toUpperCase()}: ${message.content}`).join('\n\n');
}

// CODEx: Handle local LM Studio/Ollama requests with context retries and fallbacks.
async function callLocalModel({
  type,
  endpoint,
  model,
  messages,
  temperature,
  maxTokens,
  timeoutMs,
  onStream
}) { // CODEx: Support streaming callbacks for local backends.
  if (isEmbeddingModelId(model)) { // CODEx: Guard against embedding-only checkpoints.
    throw new Error(`Model ${model} does not support text generation`); // CODEx: Block embedding checkpoints from chat usage.
  }
  const providerLabel = type === 'ollama' ? 'Ollama (Local)' : 'LM Studio'; // CODEx: Identify provider for telemetry messaging.
  let numCtx = getProviderContextLimit(type === 'ollama' ? 'ollama' : 'lmstudio'); // CODEx: Initialize local context budget.
  let attempt = 0; // CODEx: Track retry iteration count.
  const maxAttempts = 3; // CODEx: Allow retries for watchdog and context recovery.
  const chatEndpoint = resolveLocalChatEndpoint(endpoint, type, model); // CODEx: Normalize chat endpoint per provider rules.
  const reasoningMode = isReasoningModelId(model); // CODEx: Drive watchdog budget from reasoning detector.
  const deadline = Date.now() + Math.max(timeoutMs, 1000); // CODEx: Track overall timeout window.

  async function executeAttempt() {
    attempt += 1; // CODEx: Increment attempt counter for diagnostics.
    const isOllamaLocal = type === 'ollama'; // CODEx: Flag Ollama routing for payload construction.
    const payload = isOllamaLocal
      ? {
          model,
          prompt: collapseMessagesToPrompt(messages),
          stream: false,
          options: {
            temperature,
            num_predict: Math.max(256, Math.min(maxTokens || 1024, numCtx))
          }
        }
      : {
          model,
          messages: normalizeLocalMessages(messages),
          temperature,
          stream: false,
          num_ctx: numCtx,
          num_predict: Math.max(256, Math.min(maxTokens || 1024, numCtx))
        };
    if (!isOllamaLocal) {
      payload.use_default_prompt_template = false; // CODEx: Disable LM Studio Jinja templates for raw JSON payloads.
      payload.input_format = 'json'; // CODEx: Request LM Studio to interpret payload without prompt templating.
    }

    const remaining = Math.max(0, deadline - Date.now()); // CODEx: Derive remaining overall budget in milliseconds.
    if (remaining <= 0) { // CODEx: Abort when the global timeout has elapsed.
      const timeoutError = new Error(`Local request timed out after ${Math.round(timeoutMs / 1000)}s.`); // CODEx: Emit when budget exhausted.
      timeoutError.code = 'timeout'; // CODEx: Tag timeout errors for upstream handling.
      throw timeoutError; // CODEx: Surface timeout to caller.
    }

    const controller = new AbortController(); // CODEx: Manage abort lifecycle per attempt.
    let watchdogTriggered = false; // CODEx: Track watchdog intervention state.
    const watchdogLimit = Math.min(remaining, reasoningMode ? 180000 : 60000); // CODEx: Adaptive watchdog budget.
    const watchdog = setTimeout(() => {
      watchdogTriggered = true; // CODEx: Flag that the watchdog triggered.
      controller.abort(); // CODEx: Abort the pending fetch to trigger retry.
      console.warn(`[Watchdog] ${model} exceeded reasoning timeout; retrying`); // CODEx: Surface watchdog intervention.
    }, watchdogLimit); // CODEx: Schedule watchdog abort.
    const globalTimer = setTimeout(() => controller.abort(), remaining); // CODEx: Preserve overall timeout budget.

    try {
      const response = await fetch(chatEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal
      });
      clearTimeout(watchdog); // CODEx: Dispose watchdog timer after completion.
      clearTimeout(globalTimer); // CODEx: Dispose global timeout timer after completion.
      if (response.ok) {
        const data = await response.json();
        const text = isOllamaLocal
          ? data?.response || data?.output || data?.message?.content || ''
          : data?.message?.content || data?.output || data?.response || '';
        if (isOllamaLocal && attempt === 1) {
          console.log('[RAG] Provider = Ollama Local → /api/generate OK'); // CODEx: Confirm successful local routing.
        }
        if (typeof onStream === 'function') {
          onStream({ type: 'content', text });
        }
        return { content: text, reasoning: data?.message?.reasoning || '', truncated: false };
      }
      const errorText = await response.text(); // CODEx: Capture response body for diagnostics.
      if (shouldRetryStatus(response.status) && attempt < maxAttempts) {
        const delayMs = computeBackoffDelay(attempt, 600); // CODEx: Exponential backoff for transient server errors.
        console.warn(`[Bridge] ${providerLabel} HTTP ${response.status}; retrying in ${delayMs}ms (attempt ${attempt}/${maxAttempts}).`); // CODEx: Surface provider retry telemetry.
        await wait(delayMs); // CODEx: Delay subsequent attempt to respect backoff.
        return executeAttempt(); // CODEx: Re-run local request after cooldown.
      }
      if (response.status >= 400 && response.status < 500) {
        if (/no context/i.test(errorText) && attempt < maxAttempts) {
          numCtx = Math.max(512, Math.floor(numCtx * 0.7)); // CODEx: Reduce context window on capacity errors.
          return executeAttempt();
        }
        if (/model not found/i.test(errorText) && type === 'ollama') {
          const fallbackEndpoint = `${stripTrailingSlashes(deriveBaseUrl(endpoint))}/api/generate`;
          console.warn(`[Bridge] ${providerLabel} fallback to ${fallbackEndpoint} after HTTP ${response.status}.`); // CODEx: Log explicit provider fallback routing.
          const fallbackResponse = await fetch(fallbackEndpoint, { // CODEx: Retry using canonical generate endpoint.
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model,
              prompt: collapseMessagesToPrompt(messages),
              stream: false
            }),
            signal: controller.signal
          });
          if (!fallbackResponse.ok) {
            throw new Error(`HTTP ${fallbackResponse.status}: ${await fallbackResponse.text()}`);
          }
          const fallbackData = await fallbackResponse.json(); // CODEx: Decode fallback response payload.
          const fallbackText = fallbackData?.response || fallbackData?.output || ''; // CODEx: Normalize fallback text content.
          if (typeof onStream === 'function') {
            onStream({ type: 'content', text: fallbackText });
          }
          return { content: fallbackText, reasoning: '', truncated: false };
        }
      }
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    } catch (error) {
      clearTimeout(watchdog); // CODEx: Ensure watchdog timer cleared on error path.
      clearTimeout(globalTimer); // CODEx: Ensure global timer cleared on error path.
      if (error.name === 'AbortError') {
        if (watchdogTriggered && attempt < maxAttempts) {
          return executeAttempt(); // CODEx: Retry automatically after watchdog intervention.
        }
        const timeoutError = new Error(`Local request timed out after ${Math.round(timeoutMs / 1000)}s.`);
        timeoutError.code = 'timeout';
        throw timeoutError;
      }
      if (attempt < maxAttempts && isTransientNetworkError(error)) {
        const delayMs = computeBackoffDelay(attempt, 600); // CODEx: Compute exponential backoff for transient local failures.
        console.warn(`[Bridge] ${providerLabel} transient error (${error.message || error}); retrying in ${delayMs}ms.`); // CODEx: Log retry decision for diagnostics.
        await wait(delayMs); // CODEx: Pause before retrying to give the server breathing room.
        return executeAttempt(); // CODEx: Attempt the request again after backoff interval.
      }
      throw error;
    }
  }

  return executeAttempt();
}

// CODEx: Consume OpenAI-style streaming responses and merge text/reasoning segments.
async function consumeOpenAiStream(response, options = {}) {
  const { onContent, onReasoning } = options; // CODEx: Optional callbacks for incremental rendering.
  const reader = response.body?.getReader();
  if (!reader) {
    const fallback = await response.text();
    return { text: fallback, reasoning: '', truncated: false };
  }
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  const textParts = [];
  const reasoningParts = [];
  let runningText = ''; // CODEx: Track incremental content for streaming updates.
  let runningReasoning = ''; // CODEx: Track incremental reasoning for streaming updates.
  let done = false;
  while (!done) {
    const { value, done: streamDone } = await reader.read();
    done = streamDone;
    if (value) {
      buffer += decoder.decode(value, { stream: !done });
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() ?? '';
      for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line || line === 'data:' || line === 'data: [DONE]') {
          continue;
        }
        const payloadText = line.startsWith('data:') ? line.slice(5).trim() : line;
        if (payloadText === '[DONE]') {
          done = true;
          break;
        }
        try {
          const chunk = JSON.parse(payloadText);
          const delta = chunk?.choices?.[0]?.delta ?? {};
          if (typeof delta.content === 'string') {
            textParts.push(delta.content);
            runningText += delta.content; // CODEx: Build incremental assistant response.
            if (typeof onContent === 'function') {
              onContent(runningText); // CODEx: Emit partial assistant text.
            }
          }
          if (typeof delta.reasoning === 'string') {
            reasoningParts.push(delta.reasoning);
            runningReasoning += delta.reasoning; // CODEx: Build incremental reasoning trace.
            if (typeof onReasoning === 'function') {
              onReasoning(runningReasoning); // CODEx: Emit partial reasoning text.
            }
          }
          if (Array.isArray(delta.reasoning)) {
            for (const item of delta.reasoning) {
              if (typeof item?.text === 'string') {
                reasoningParts.push(item.text);
                runningReasoning += item.text; // CODEx: Extend reasoning accumulation for array payloads.
                if (typeof onReasoning === 'function') {
                  onReasoning(runningReasoning); // CODEx: Emit aggregated reasoning updates.
                }
              }
            }
          }
        } catch (error) {
          textParts.push(payloadText);
          runningText += payloadText; // CODEx: Append raw payload fallback.
          if (typeof onContent === 'function') {
            onContent(runningText); // CODEx: Emit fallback chunk.
          }
        }
      }
    }
  }
  return {
    text: textParts.join(''),
    reasoning: reasoningParts.join('\n').trim(),
    truncated: false
  };
}

function trimResponseToTokenLimit(text, limit) {
  if (!text || !Number.isFinite(limit) || limit <= 0) {
    return { content: text ?? '', truncated: false };
  }

  const estimated = estimateTokenCount(text);
  if (estimated <= limit) {
    return { content: text, truncated: false };
  }

  const segments = text.split(/(\s+)/);
  let remaining = limit;
  let result = '';
  for (const segment of segments) {
    if (!segment.trim()) {
      result += segment;
      continue;
    }
    const weight = Math.max(1, Math.ceil(segment.trim().length / 4));
    if (weight > remaining) {
      break;
    }
    result += segment;
    remaining -= weight;
  }

  if (!result.trim()) {
    const maxChars = Math.max(16, Math.round(limit * 4));
    const sliced = text.slice(0, maxChars).trim();
    const decorated = `${sliced} … [truncated to ~${limit} tokens]`;
    return { content: decorated, truncated: true };
  }

  if (!/\s$/.test(result)) {
    result += ' ';
  }
  result += `… [truncated to ~${limit} tokens]`;
  return { content: result.trimEnd(), truncated: true };
}

function normalizeModelMessage(message) {
  const textParts = [];
  const reasoningParts = [];
  const push = (value, target) => {
    if (typeof value !== 'string') return;
    const trimmed = value.trim();
    if (trimmed) {
      target.push(trimmed);
    }
  };

  if (!message) {
    return { text: '', reasoning: '' };
  }

  const content = message.content;
  if (Array.isArray(content)) {
    for (const chunk of content) {
      if (chunk == null) continue;
      if (typeof chunk === 'string') {
        push(chunk, textParts);
        continue;
      }
      const type = chunk.type ?? '';
      if (type === 'text' || type === 'output_text') {
        push(chunk.text ?? chunk.value ?? '', textParts);
      } else if (type === 'reasoning' || type === 'reasoning_output') {
        push(chunk.reasoning ?? chunk.text ?? chunk.value ?? '', reasoningParts);
      } else if (type === 'tool_call' && chunk.output) {
        push(chunk.output, textParts);
      } else if (chunk.text) {
        push(chunk.text, textParts);
      }
    }
  } else if (typeof content === 'string') {
    push(content, textParts);
  }

  const reasoning = message.reasoning;
  if (Array.isArray(reasoning)) {
    for (const item of reasoning) {
      if (!item) continue;
      if (typeof item === 'string') {
        push(item, reasoningParts);
      } else {
        push(item.text ?? item.value ?? item.reasoning ?? '', reasoningParts);
      }
    }
  } else if (typeof reasoning === 'string') {
    push(reasoning, reasoningParts);
  } else if (reasoning && typeof reasoning === 'object') {
    push(reasoning.text ?? reasoning.value ?? '', reasoningParts);
  }

  if (message.metadata?.reasoning) {
    push(message.metadata.reasoning, reasoningParts);
  }

  return {
    text: textParts.join('\n\n'),
    reasoning: reasoningParts.join('\n\n')
  };
}

async function consumeTextStream(response, options = {}) {
  const { onContent } = options; // CODEx: Allow streaming updates for chunked JSON/text payloads.
  const reader = response.body?.getReader();
  if (!reader) {
    const fallback = await response.text();
    if (typeof onContent === 'function') {
      onContent(fallback); // CODEx: Emit fallback content when streaming is unavailable.
    }
    return { text: fallback, reasoning: '', truncated: false };
  }
  const decoder = new TextDecoder('utf-8');
  let aggregated = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    if (value) {
      aggregated += decoder.decode(value, { stream: true }); // CODEx: Accumulate streamed chunk.
      if (typeof onContent === 'function') {
        onContent(aggregated); // CODEx: Push incremental buffer to UI.
      }
    }
  }
  aggregated += decoder.decode(); // CODEx: Flush decoder state.
  if (typeof onContent === 'function') {
    onContent(aggregated); // CODEx: Emit finalized aggregation for completeness.
  }
  return { text: aggregated, reasoning: '', truncated: false };
}

function resolveRequestTimeout(modelId, preset, overrideTimeout) {
  if (Number.isFinite(overrideTimeout) && overrideTimeout > 0) {
    return overrideTimeout;
  }

  const requestSeconds = Number.isFinite(config?.requestTimeoutSeconds)
    ? config.requestTimeoutSeconds
    : defaultConfig.requestTimeoutSeconds;
  const reasoningSeconds = Number.isFinite(config?.reasoningTimeoutSeconds)
    ? config.reasoningTimeoutSeconds
    : defaultConfig.reasoningTimeoutSeconds;

  const baseMs = Math.max(1000, Math.round(requestSeconds * 1000));
  let reasoningMs = Math.max(baseMs, Math.round(reasoningSeconds * 1000));
  reasoningMs = Math.max(reasoningMs, 300000);

  if (preset?.reasoningTimeoutMs && Number.isFinite(preset.reasoningTimeoutMs) && preset.reasoningTimeoutMs > 0) {
    reasoningMs = Math.max(reasoningMs, preset.reasoningTimeoutMs);
  }

  const reasoningActive = isReasoningModeActive({
    model: modelId,
    providerPreset: preset?.id ?? config.providerPreset
  });

  if (reasoningActive) {
    return reasoningMs;
  }

  return baseMs;
}

async function callModel(messages, overrides = {}) {
  const endpoint = overrides.endpoint ?? config.endpoint;
  let model = overrides.model ?? config.model; // CODEx: Allow reassignment for embedding guardrails.
  const temperature = overrides.temperature ?? config.temperature;
  const apiKey = overrides.apiKey ?? config.apiKey;
  const providerPreset = overrides.providerPreset ?? config.providerPreset;
  const openRouterPolicy = (overrides.openRouterPolicy ?? config.openRouterPolicy ?? '').trim();

  if (!endpoint || !model) {
    throw new Error('Model endpoint or name missing.');
  }

  const preset = providerPresetMap.get(providerPreset);
  model = ensureChatModel(model, preset); // CODEx: Prevent embedding checkpoints from entering chat flows.
  const timeoutMs = resolveRequestTimeout(model, preset, overrides.timeoutMs);
  const connectionType = detectConnectionType(endpoint, preset);
  const reasoningActive = isReasoningModeActive({ model, providerPreset });
  const streamCallback = typeof overrides.onStream === 'function' ? overrides.onStream : null; // CODEx: Capture optional streaming hook.
  const contextLimit = getProviderContextLimit(providerPreset);
  let effectiveMaxTokens = overrides.maxTokens ?? config.maxResponseTokens ?? contextLimit;
  if (!Number.isFinite(effectiveMaxTokens) || effectiveMaxTokens <= 0) {
    effectiveMaxTokens = contextLimit;
  }
  if (reasoningActive) {
    const reasoningCeiling = Math.min(contextLimit, 6000); // CODEx: Cap reasoning responses to a 6k token ceiling.
    const reasoningFloor = Math.min(reasoningCeiling, 4500); // CODEx: Ensure at least 4.5k tokens when possible.
    effectiveMaxTokens = Math.max(effectiveMaxTokens, reasoningFloor); // CODEx: Lift lower bound for reasoning mode.
    effectiveMaxTokens = Math.min(effectiveMaxTokens, reasoningCeiling); // CODEx: Guard against runaway token budgets.
  }
  effectiveMaxTokens = Math.min(contextLimit, effectiveMaxTokens);
  let plannedTokens = effectiveMaxTokens; // CODEx: Track requested token budget for telemetry.

  const callStarted = getTimestampMs();
  // CODEx: Collect model latency metrics for diagnostics output.
  const finalizeMetrics = () => {
    const duration = Math.max(0, getTimestampMs() - callStarted);
    modelCallMetrics.totalMs += duration;
    modelCallMetrics.count += 1;
  };
  const logReasoningDiagnostics = () => {
    if (!reasoningActive) {
      return; // CODEx: Only emit telemetry for reasoning-enabled runs.
    }
    const turnDuration = Math.max(0, getTimestampMs() - callStarted); // CODEx: Derive latency for reporting.
    console.table({
      model, // CODEx: Report model identifier.
      reasoningMode: reasoningActive, // CODEx: Confirm reasoning handshake state.
      tokensUsed: plannedTokens, // CODEx: Surface negotiated token budget.
      turnDuration, // CODEx: Highlight runtime in milliseconds.
      entropyScore: Number.isFinite(currentEntropyScore)
        ? Number(currentEntropyScore.toFixed(3))
        : null, // CODEx: Snapshot current entropy loop score.
      reflexTriggered: Boolean(lastReflexSummaryAt && lastReflexSummaryAt >= callStarted) // CODEx: Flag reflex overlap.
    });
  }; // CODEx: Emit structured diagnostics for debugging exports.

  if (connectionType !== 'openai') {
    try {
      const result = await callLocalModel({
        type: connectionType,
        endpoint,
        model,
        messages,
        temperature,
        maxTokens: effectiveMaxTokens,
        timeoutMs,
        onStream: streamCallback
      });
      finalizeMetrics();
      logReasoningDiagnostics(); // CODEx: Mirror remote telemetry for local providers.
      return result;
    } catch (error) {
      finalizeMetrics();
      if (reasoningActive && (error?.code === 'timeout' || /timed out/i.test(error?.message ?? ''))) {
        modelCallMetrics.reasoningTimeouts += 1;
      }
      throw error;
    }
  }

  const isAnthropic = preset?.anthropicFormat;
  const isGoogle = preset?.googleFormat;
  const usingOpenRouter = (preset?.id ?? providerPreset) === 'openrouter' || /openrouter\.ai/.test(endpoint);
  const shouldStream = overrides.stream === true || (overrides.stream !== false && reasoningActive && !isAnthropic && !isGoogle);

  let headers = { 'Content-Type': 'application/json' };
  let payload = {};
  let requestEndpoint = endpoint;

  if (isAnthropic) {
    headers = {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01'
    };
    payload = {
      model,
      max_tokens: Math.round(effectiveMaxTokens) || 4096,
      temperature,
      messages: messages.map((msg) => ({
        role: msg.role === 'assistant' ? 'assistant' : 'user',
        content: msg.content
      }))
    };
    plannedTokens = payload.max_tokens; // CODEx: Record anthropic token ceiling for telemetry.
  } else if (isGoogle) {
    requestEndpoint = `${endpoint}${model}:generateContent?key=${apiKey}`;
    payload = {
      contents: [{
        parts: [{ text: messages.map((msg) => `${msg.role}: ${msg.content}`).join('\n\n') }]
      }],
      generationConfig: {
        temperature,
        maxOutputTokens: Math.round(effectiveMaxTokens) || 2048
      }
    };
    plannedTokens = payload.generationConfig.maxOutputTokens; // CODEx: Track Google token plan for telemetry output.
  } else {
    if (apiKey) {
      headers.Authorization = `Bearer ${apiKey}`;
    }
    if (usingOpenRouter && openRouterPolicy) {
      headers['X-OpenRouter-Data-Policy'] = openRouterPolicy;
    }
    payload = {
      model,
      messages,
      temperature,
      stream: shouldStream
    };
    if (Number.isFinite(effectiveMaxTokens) && effectiveMaxTokens > 0) {
      payload.max_tokens = Math.round(effectiveMaxTokens);
      plannedTokens = payload.max_tokens; // CODEx: Persist adjusted max token budget for diagnostics.
    }
  }

  const maxAttempts = 3; // CODEx: Allow exponential backoff retries for transient failures.
  let attempt = 0; // CODEx: Track remote retry attempts.

  async function issueRequest() {
    attempt += 1; // CODEx: Increment attempt counter for diagnostics.
    const controller = new AbortController(); // CODEx: Manage fetch cancellation per attempt.
    let watchdogTriggered = false; // CODEx: Track watchdog activation state.
    const watchdogLimit = reasoningActive ? 180000 : 60000; // CODEx: Adaptive watchdog threshold.
    const watchdogTimer = setTimeout(() => {
      watchdogTriggered = true; // CODEx: Record watchdog activation.
      controller.abort(); // CODEx: Abort fetch to enable retry.
      console.warn(`[Watchdog] ${model} exceeded reasoning timeout; retrying`); // CODEx: Surface watchdog telemetry.
    }, watchdogLimit); // CODEx: Schedule watchdog for long requests.
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs); // CODEx: Preserve configured timeout budget.
    try {
      const response = await fetch(requestEndpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal
      });
      clearTimeout(timeoutId); // CODEx: Dispose configured timeout timer.
      clearTimeout(watchdogTimer); // CODEx: Dispose watchdog timer after completion.
      return { response, watchdogTriggered }; // CODEx: Return response with watchdog flag.
    } catch (error) {
      clearTimeout(timeoutId); // CODEx: Ensure timers cleared on error path.
      clearTimeout(watchdogTimer); // CODEx: Ensure watchdog timer cleared on error path.
      if (error.name === 'AbortError') {
        if (watchdogTriggered) {
          error.code = 'watchdog-abort'; // CODEx: Flag watchdog-driven abort for backoff handling.
          throw error; // CODEx: Bubble to outer retry controller.
        }
        if (reasoningActive) {
          modelCallMetrics.reasoningTimeouts += 1; // CODEx: Track reasoning timeout metrics.
        }
        const seconds = Math.round(timeoutMs / 100) / 10; // CODEx: Convert timeout to seconds with decimal precision.
        const timeoutError = new Error(`Request timed out after ${seconds}s.`); // CODEx: Provide human-readable timeout error.
        timeoutError.code = 'timeout'; // CODEx: Tag timeout errors for upstream handling.
        throw timeoutError;
      }
      throw error; // CODEx: Propagate non-timeout errors.
    }
  }

  let response;
  let lastError = null; // CODEx: Track last failure for final error reporting.
  while (attempt < maxAttempts) {
    try {
      const { response: resolvedResponse } = await issueRequest(); // CODEx: Execute request with watchdog handling.
      response = resolvedResponse; // CODEx: Normalize response reference for downstream logic.
    } catch (error) {
      lastError = error; // CODEx: Preserve error for diagnostics.
      if (attempt < maxAttempts && (error.code === 'watchdog-abort' || isTransientNetworkError(error))) {
        const delayMs = computeBackoffDelay(attempt, 600); // CODEx: Determine exponential backoff interval.
        console.warn(`[Bridge] ${model} retry after ${error.code === 'watchdog-abort' ? 'watchdog abort' : error.message}; waiting ${delayMs}ms.`); // CODEx: Emit retry telemetry for remote providers.
        await wait(delayMs); // CODEx: Pause before attempting again.
        continue; // CODEx: Retry request after cooldown.
      }
      finalizeMetrics(); // CODEx: Preserve metrics on terminal failure.
      throw error; // CODEx: Propagate non-retriable error to caller.
    }
    if (!response) {
      break; // CODEx: Should not occur, but guard to avoid infinite loop.
    }
    if (!response.ok) {
      if (attempt < maxAttempts && shouldRetryStatus(response.status)) {
        const delayMs = computeBackoffDelay(attempt, 600); // CODEx: Compute retry interval for server failures.
        console.warn(`[Bridge] HTTP ${response.status} at ${requestEndpoint}; retrying in ${delayMs}ms (attempt ${attempt}/${maxAttempts}).`); // CODEx: Log remote fallback path clearly.
        await wait(delayMs); // CODEx: Delay before reissuing request.
        continue; // CODEx: Attempt request again following backoff.
      }
    }
    break; // CODEx: Exit loop when response obtained (success or terminal failure).
  }

  if (!response) {
    finalizeMetrics(); // CODEx: Preserve metrics when no response obtained.
    throw lastError || new Error('Model request failed without response.'); // CODEx: Report absence of response clearly.
  }

  const contentType = response.headers.get('content-type') ?? ''; // CODEx: Capture response type for streaming selection.

  if (!response.ok) {
    const text = await response.text();
    finalizeMetrics();
    throw new Error(`HTTP ${response.status}: ${text}`);
  }

  if (shouldStream && response.body) {
    if (contentType.includes('text/event-stream')) {
      const streamed = await consumeOpenAiStream(response, {
        onContent: (text) => {
          if (streamCallback) {
            streamCallback({ type: 'content', text }); // CODEx: Surface incremental assistant text.
          }
        },
        onReasoning: (text) => {
          if (streamCallback) {
            streamCallback({ type: 'reasoning', text }); // CODEx: Surface incremental reasoning trace.
          }
        }
      });
      finalizeMetrics();
      logReasoningDiagnostics();
      return { content: streamed.text, truncated: streamed.truncated, reasoning: streamed.reasoning };
    }
    const streamed = await consumeTextStream(response, {
      onContent: (text) => {
        if (streamCallback) {
          streamCallback({ type: 'content', text }); // CODEx: Relay chunked text buffers for non-SSE streaming.
        }
      }
    });
    finalizeMetrics();
    logReasoningDiagnostics();
    return { content: streamed.text, truncated: streamed.truncated ?? false, reasoning: streamed.reasoning ?? '' };
  }

  const data = await response.json();

  let message = {};
  if (isAnthropic) {
    message = {
      content: data.content?.[0]?.text || '',
      reasoning: data.content?.find((c) => c.type === 'thinking')?.thinking || ''
    };
  } else if (isGoogle) {
    message = {
      content: data.candidates?.[0]?.content?.parts?.[0]?.text || ''
    };
  } else {
    message = data?.choices?.[0]?.message ?? {};
  }

  const { text, reasoning } = normalizeModelMessage(message);
  if (streamCallback) {
    streamCallback({ type: 'content', text }); // CODEx: Emit final consolidated content for UI sync.
    if (reasoning) {
      streamCallback({ type: 'reasoning', text: reasoning }); // CODEx: Share reasoning payload when available.
    }
  }
  const segments = [];
  if (reasoning) {
    segments.push(`Reasoning:\n${reasoning}`);
  }
  if (text) {
    segments.push(text);
  }
  const combined = segments.join('\n\n').trim();
  if (!combined) {
    finalizeMetrics();
    logReasoningDiagnostics();
    return { content: '', truncated: false, reasoning: reasoning ?? '' };
  }

  if (!Number.isFinite(effectiveMaxTokens) || effectiveMaxTokens <= 0) {
    finalizeMetrics();
    logReasoningDiagnostics();
    return { content: combined, truncated: false, reasoning };
  }

  const trimmedResult = trimResponseToTokenLimit(combined, Math.round(effectiveMaxTokens));
  finalizeMetrics();
  logReasoningDiagnostics();
  return { ...trimmedResult, reasoning };
}

function toggleVoiceInput() {
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    alert('Speech recognition is not supported in this browser.');
    return;
  }

  if (!recognition) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      elements.messageInput.value = transcript;
      elements.voiceButton.setAttribute('aria-pressed', 'false');
      isVoiceActive = false;
      void handleUserMessage();
    };
    recognition.onerror = () => {
      elements.voiceButton.setAttribute('aria-pressed', 'false');
      isVoiceActive = false;
    };
    recognition.onend = () => {
      elements.voiceButton.setAttribute('aria-pressed', 'false');
      isVoiceActive = false;
    };
  }

  if (isVoiceActive) {
    recognition.stop();
    isVoiceActive = false;
    elements.voiceButton.setAttribute('aria-pressed', 'false');
  } else {
    recognition.start();
    isVoiceActive = true;
    elements.voiceButton.setAttribute('aria-pressed', 'true');
  }
}

async function speakText(text, { force = false } = {}) {
  const phrase = typeof text === 'string' ? text.trim() : String(text ?? '').trim();
  if (!phrase) return;
  if (!force && !config.autoSpeak) return;

  const preset = getTtsPreset() ?? getTtsPreset(defaultConfig.ttsPreset);
  if (!preset || preset.provider === 'browser') {
    speakWithBrowserVoice(phrase);
    return;
  }

  const hasWebAudio = Boolean(window.AudioContext || window.webkitAudioContext);
  if (!hasWebAudio) {
    console.warn('Web Audio API unavailable; falling back to browser voice.');
    speakWithBrowserVoice(phrase);
    return;
  }

  try {
    const audioBlob = await synthesizeRemoteTts(phrase, preset);
    if (!audioBlob) {
      throw new Error('TTS server returned no audio data.');
    }
    await playAudioBlob(audioBlob);
  } catch (error) {
    console.error('TTS synthesis failed', error);
    notifyTtsError(error, preset);
    speakWithBrowserVoice(phrase);
  }
}

function speakWithBrowserVoice(text) {
  if (!('speechSynthesis' in window)) {
    console.warn('Speech synthesis is not supported in this browser.');
    return;
  }
  const utterance = new SpeechSynthesisUtterance(text);
  const desiredVoice = config.ttsVoiceId?.trim();
  if (desiredVoice) {
    const voices = window.speechSynthesis.getVoices();
    const voice = voices.find((item) => item.name === desiredVoice || item.voiceURI === desiredVoice);
    if (voice) {
      utterance.voice = voice;
    }
  }
  utterance.volume = getVolumeScalar();
  window.speechSynthesis.speak(utterance);
}

async function synthesizeRemoteTts(text, preset) {
  const baseUrl = (config.ttsServerUrl || preset.defaultUrl || '').trim();
  const voiceId = (config.ttsVoiceId || preset.voiceId || '').trim();
  const headers = { 'Content-Type': 'application/json' };
  const payload = { text };
  let targetUrl = baseUrl;

  if (preset.needsApiKey && preset.provider !== 'elevenlabs') {
    if (!config.ttsApiKey) {
      throw new Error('Provide the API key required by this TTS service.');
    }
    headers.Authorization = `Bearer ${config.ttsApiKey}`;
  }

  switch (preset.provider) {
    case 'piper': {
      if (!baseUrl) throw new Error('Set the Piper server URL.');
      targetUrl = joinUrl(baseUrl, preset.endpoint ?? '/synthesize');
      payload.voice = voiceId || preset.voiceId;
      payload.output_format = 'wav';
      payload.sample_rate = preset.sampleRate ?? 22050;
      break;
    }
    case 'coqui_xtts': {
      if (!baseUrl) throw new Error('Set the Coqui XTTS server URL.');
      targetUrl = joinUrl(baseUrl, preset.endpoint ?? '/api/tts');
      payload.speaker = voiceId || preset.voiceId;
      payload.language = preset.language ?? 'en';
      payload.stream = false;
      break;
    }
    case 'mimic3': {
      if (!baseUrl) throw new Error('Set the Mimic 3 server URL.');
      targetUrl = joinUrl(baseUrl, preset.endpoint ?? '/api/tts');
      payload.voice = voiceId || preset.voiceId;
      payload.audio_format = 'wav';
      break;
    }
    case 'bark': {
      if (!baseUrl) throw new Error('Set the Bark server URL.');
      targetUrl = joinUrl(baseUrl, preset.endpoint ?? '/generate');
      payload.voice = voiceId || preset.voiceId;
      payload.format = 'wav';
      break;
    }
    case 'f5': {
      if (!baseUrl) throw new Error('Set the F5-TTS server URL.');
      targetUrl = joinUrl(baseUrl, preset.endpoint ?? '/api/tts');
      payload.voice = voiceId || preset.voiceId;
      payload.format = 'wav';
      break;
    }
    case 'elevenlabs': {
      if (!baseUrl) throw new Error('Set the ElevenLabs base URL.');
      if (!voiceId) throw new Error('Provide an ElevenLabs voice ID.');
      if (!config.ttsApiKey) throw new Error('Provide your ElevenLabs API key.');
      headers['xi-api-key'] = config.ttsApiKey;
      delete headers.Authorization;
      targetUrl = joinUrl(baseUrl, voiceId);
      return requestAudioBlob(targetUrl, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          text,
          model_id: preset.modelId ?? 'eleven_monolingual_v1',
          voice_settings: { stability: 0.5, similarity_boost: 0.5 }
        })
      });
    }
    default: {
      if (!baseUrl) throw new Error('Configure the TTS server URL for this preset.');
      payload.voice = voiceId;
    }
  }

  return requestAudioBlob(targetUrl, {
    method: preset.method ?? 'POST',
    headers,
    body: JSON.stringify(payload)
  });
}

async function requestAudioBlob(url, options) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 15000); // 15s timeout for TTS

  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(timeoutId);
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`${response.status} ${response.statusText}: ${errorText}`.trim());
    }
    const contentType = response.headers.get('content-type') ?? '';
    if (contentType.includes('application/json')) {
      const data = await response.json();
      if (data.audio) {
        return base64ToBlob(data.audio, data.mime_type ?? data.format ?? 'audio/wav');
      }
      if (data.url) {
        const followUp = await fetch(data.url);
        if (!followUp.ok) {
          throw new Error(`Failed to download audio from ${data.url}`);
        }
        return followUp.blob();
      }
      throw new Error('TTS server returned JSON without audio payload.');
    }
    return response.blob();
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error('TTS request timed out after 15 seconds.');
    }
    throw error;
  }
}

async function playAudioBlob(blob) {
  const context = getAudioContext();
  if (!context) {
    console.warn('Audio playback unavailable in this browser.');
    return;
  }
  if (context.state === 'suspended') {
    await context.resume();
  }
  const arrayBuffer = await blob.arrayBuffer();
  const audioBuffer = await new Promise((resolve, reject) => {
    context.decodeAudioData(arrayBuffer.slice(0), resolve, reject);
  });
  const source = context.createBufferSource();
  source.buffer = audioBuffer;
  const gainNode = context.createGain();
  gainNode.gain.value = getVolumeScalar();
  source.connect(gainNode).connect(context.destination);
  source.start();
}

function notifyTtsError(error, preset) {
  const now = Date.now();
  if (now - lastTtsErrorAt < TTS_ERROR_COOLDOWN) return;
  lastTtsErrorAt = now;
  addSystemMessage(`TTS fallback (${preset.label}): ${error.message}`);
}

function getAudioContext() {
  if (audioContext) return audioContext;
  const AudioContextConstructor = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextConstructor) return undefined;
  audioContext = new AudioContextConstructor();
  return audioContext;
}

function base64ToBlob(base64, mimeType = 'audio/wav') {
  const cleaned = base64.replace(/^data:.*;base64,/, '');
  const binary = window.atob(cleaned);
  const length = binary.length;
  const bytes = new Uint8Array(length);
  for (let index = 0; index < length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Blob([bytes], { type: mimeType });
}

function joinUrl(base, path) {
  if (!base) return path || '';
  if (!path) return base;
  const normalizedBase = base.endsWith('/') ? base.slice(0, -1) : base;
  const normalizedPath = path.startsWith('/') ? path.slice(1) : path;
  return `${normalizedBase}/${normalizedPath}`;
}

function getVolumeScalar() {
  return clampNumber(config.ttsVolume ?? defaultConfig.ttsVolume, 0, 200, defaultConfig.ttsVolume) / 100;
}

function clearChatWindow() {
  elements.chatWindow.innerHTML = '';
}

async function exportConversation() {
  const all = await getAllMessages();
  const content = all
    .sort((a, b) => a.timestamp - b.timestamp)
    .map((entry) => {
      const turn = entry.turnNumber ? `Turn ${entry.turnNumber}` : '';
      return `${new Date(normalizeTimestamp(entry.timestamp)).toISOString()}\t${entry.role}\t${turn}\t${entry.content}`;
    })
    .join('\n');
  const blob = new Blob([content], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `sam-conversation-${new Date().toISOString().slice(0, 10)}.txt`;
  anchor.click();
  URL.revokeObjectURL(url);
}

async function startDualChat() {
  if (isDualChatRunning) {
    addSystemMessage('Dual chat already running. Stop it before starting again.');
    return;
  }
  if (!config.agentBEnabled) {
    addSystemMessage('Enable Model B in options before starting the arena conversation.');
    return;
  }
  const agentAConnection = getAgentConnection('A');
  const agentBConnection = getAgentConnection('B');
  if (!agentAConnection.endpoint || !agentAConnection.model) {
    addSystemMessage(`${getAgentDisplayName('A')} is missing an endpoint or model. Configure the connection before starting.`);
    updateModelConnectionStatus();
    return;
  }
  if (!agentBConnection.endpoint || !agentBConnection.model) {
    addSystemMessage(`${getAgentDisplayName('B')} is missing an endpoint or model. Configure the connection before starting.`);
    updateModelConnectionStatus();
    return;
  }
  const providedSeed = elements.dualSeedInput.value.trim();
  const seed = providedSeed || config.dualSeed || DEFAULT_DUAL_SEED;
  if (!providedSeed) {
    elements.dualSeedInput.value = seed;
    config.dualSeed = seed;
    saveConfig();
  }

  activeDualSeed = seed;
  rotateRagSession(MODE_ARENA);
  resetDualChat();
  clearDualCountdownTimer();
  activeDualConnections.A = agentAConnection;
  activeDualConnections.B = agentBConnection;
  isDualChatRunning = true;
  dualTurnsCompleted = 0;
  const truncatedSeed = seed.length > 120 ? `${seed.slice(0, 117)}…` : seed;
  updateProcessState('arena', { status: 'Running', detail: 'Dual chat running.' });
  updateProcessState('modelA', {
    status: 'Preparing',
    detail: `${getAgentDisplayName('A')} is crafting the next arena turn.`
  });
  updateProcessState('modelB', {
    status: 'Listening',
    detail: `${getAgentDisplayName('B')} awaiting the opening argument.`
  });
  void recordLog('arena', `Dual chat started with seed: ${truncatedSeed}`, { silent: true });
  recordDualMessage('B', seed, {
    saveHistory: false,
    label: 'Conversation seed',
    marker: 'seed',
    turnNumber: 0
  });
  elements.dualStatus.textContent = 'Dual chat running…';
  updateModelConnectionStatus();
  startDualAutosaveTimer();
  await activatePhase(0, { force: true, announce: true });
  updateReflexStatus(config.reflexEnabled ? 'Reflex armed' : 'Reflex disabled', config.reflexEnabled ? 'ready' : 'disabled');
  updateEntropyMeter();
  await generateDualTurn('A', seed, { isInitial: true });
  if (config.dualAutoContinue) {
    scheduleNextDualTurn();
  }
}

function resetDualChat() {
  stopDualChat();
  elements.dualChatWindow.innerHTML = '';
  dualChatHistory = [];
  dualTurnCounter = 0;
  nextDualSpeaker = 'A'; // Reset to A to ensure proper starting state
  activeDualSeed = config.dualSeed ?? DEFAULT_DUAL_SEED;
  elements.dualStatus.textContent = 'Awaiting start…';
  reflexSummaryCount = 0;
  lastReflexSummaryAt = null;
  reflexInFlight = false;
  activePhaseIndex = 0;
  activePhaseId = null;
  pendingPhaseAdvance = false;
  lastEntropyState = 'calibrating';
  lastReflexSynthesisSignature = null;
  updateProcessState('arena', { status: 'Idle', detail: 'Dual chat reset.' });
  updateEntropyMeter();
  updateReflexStatus();
}

function stopDualChat() {
  isDualChatRunning = false;
  if (autoContinueTimer) {
    clearTimeout(autoContinueTimer);
    autoContinueTimer = undefined;
  }
  clearDualCountdownTimer();
  stopDualAutosaveTimer();
  activeDualConnections.A = null;
  activeDualConnections.B = null;
  elements.dualStatus.textContent = 'Dual chat stopped.';
  flushCheckpointBuffer(MODE_ARENA);
  updateModelConnectionStatus();
  void persistDualRagSnapshot();
  reflexInFlight = false;
  pendingPhaseAdvance = false;
  updateProcessState('arena', { status: 'Idle', detail: 'Dual chat stopped.' });
  updateProcessState('modelA', { status: 'Idle', detail: 'Waiting for user prompt.' });
  const modelBDetail = config.agentBEnabled ? 'Awaiting arena start.' : 'Model B is disabled.';
  updateProcessState('modelB', {
    status: config.agentBEnabled ? 'Idle' : 'Disabled',
    detail: modelBDetail
  });
  updateEntropyMeter();
  updateReflexStatus();
  void recordLog('arena', 'Dual chat stopped.', { silent: true });
}

function hasDualTurnLimit() {
  return typeof config.dualTurnLimit === 'number' && config.dualTurnLimit > 0;
}

function dualTurnLimitReached() {
  return hasDualTurnLimit() && dualTurnsCompleted >= config.dualTurnLimit;
}

async function advanceDualTurn() {
  if (!isDualChatRunning) {
    addSystemMessage('Start the dual chat before stepping through turns.');
    return;
  }
  if (dualTurnLimitReached()) {
    elements.dualStatus.textContent = `Dual chat reached turn limit of ${config.dualTurnLimit} turns.`;
    stopDualChat();
    return;
  }
  if (autoContinueTimer) {
    clearTimeout(autoContinueTimer);
    autoContinueTimer = undefined;
  }
  clearDualCountdownTimer();
  await generateDualTurn(nextDualSpeaker);
  if (config.dualAutoContinue) {
    scheduleNextDualTurn();
  }
}

function clearDualCountdownTimer() {
  if (dualCountdownTimer) {
    clearInterval(dualCountdownTimer);
    dualCountdownTimer = undefined;
  }
  dualCountdownDeadline = null;
}

function updateDualCountdownStatus() {
  if (!elements.dualStatus) return;
  if (!isDualChatRunning || dualCountdownDeadline === null) {
    if (isDualChatRunning) {
      elements.dualStatus.textContent = 'Dual chat running…';
    }
    return;
  }
  const remainingMs = dualCountdownDeadline - Date.now();
  if (remainingMs <= 0) {
    elements.dualStatus.textContent = 'Dual chat running…';
    clearDualCountdownTimer();
    return;
  }
  const seconds = Math.max(0, Math.ceil(remainingMs / 1000));
  elements.dualStatus.textContent = `Next turn in ${seconds}s…`;
}

function scheduleNextDualTurn() {
  if (!isDualChatRunning) return;
  if (dualTurnLimitReached()) {
    elements.dualStatus.textContent = `Dual chat reached turn limit of ${config.dualTurnLimit} turns.`;
    stopDualChat();
    return;
  }
  clearTimeout(autoContinueTimer);
  clearDualCountdownTimer();
  const delaySeconds = resolveDualTurnDelaySeconds();
  const delayMs = Math.max(0, Math.round(delaySeconds * 1000));
  if (delayMs === 0) {
    elements.dualStatus.textContent = 'Dual chat running…';
    autoContinueTimer = setTimeout(() => {
      autoContinueTimer = undefined;
      void safeDualTurnInvocation(); // CODEx: Route through watchdog wrapper for error recovery.
    }, 0);
    return;
  }
  dualCountdownDeadline = Date.now() + delayMs;
  updateDualCountdownStatus();
  dualCountdownTimer = setInterval(() => {
    updateDualCountdownStatus();
  }, 1000);
  autoContinueTimer = setTimeout(() => {
    autoContinueTimer = undefined;
    clearDualCountdownTimer();
    void safeDualTurnInvocation(); // CODEx: Protect auto loop from transient failures.
  }, delayMs);
}

function resolveDualTurnDelaySeconds() {
  const baseDelay = clampNumber(
    config.dualTurnDelaySeconds ?? defaultConfig.dualTurnDelaySeconds,
    0,
    600,
    defaultConfig.dualTurnDelaySeconds
  );
  const nextConnection = activeDualConnections[nextDualSpeaker] ?? getAgentConnection(nextDualSpeaker); // CODEx: Inspect next speaker handshake.
  const reasoningLoop = isReasoningModeActive({
    model: nextConnection?.model,
    providerPreset: nextConnection?.providerPreset
  }); // CODEx: Determine if adaptive reasoning delay applies.
  const entropyScore = Number.isFinite(currentEntropyScore) ? currentEntropyScore : 0; // CODEx: Normalize entropy metric.
  if (reasoningLoop) {
    const target = entropyScore > 0.8 ? 25 : 12; // CODEx: Apply Phase II adaptive windowing.
    return Math.min(45, Math.max(8, target));
  }
  if (baseDelay === 0) {
    return 0;
  }
  if (!Number.isFinite(currentEntropyScore)) {
    return baseDelay;
  }
  if (currentEntropyScore >= ENTROPY_LOOP_THRESHOLD) {
    return Math.max(MIN_DYNAMIC_DELAY_SECONDS, Math.round(baseDelay * LOOP_DELAY_SCALE));
  }
  if (currentEntropyScore <= ENTROPY_EXPLORE_THRESHOLD) {
    return Math.min(MAX_DYNAMIC_DELAY_SECONDS, Math.round(baseDelay * EXPLORE_DELAY_SCALE));
  }
  return baseDelay;
}

async function safeDualTurnInvocation() {
  if (!isDualChatRunning) {
    return; // CODEx: Avoid scheduling when arena is idle.
  }
  try {
    await advanceDualTurn();
  } catch (error) {
    console.warn('Turn crash — retrying in 5 s', error); // CODEx: Surface watchdog retry notice.
    if (!isDualChatRunning) {
      return;
    }
    if (elements.dualStatus) {
      elements.dualStatus.textContent = 'Recovering from model error…'; // CODEx: Inform operators about retry cycle.
    }
    autoContinueTimer = setTimeout(() => {
      autoContinueTimer = undefined;
      void safeDualTurnInvocation();
    }, 5000);
  }
}

function startDualAutosaveTimer() {
  stopDualAutosaveTimer();
  dualAutosaveTimer = setInterval(() => {
    if (!isDualChatRunning) return;
    void persistDualRagSnapshot();
  }, ARENA_AUTOSAVE_INTERVAL);
  updateProcessState('autosave', {
    status: 'Running',
    detail: 'Saving arena transcript every 2 minutes.'
  });
}

function stopDualAutosaveTimer() {
  if (dualAutosaveTimer) {
    clearInterval(dualAutosaveTimer);
    dualAutosaveTimer = undefined;
  }
  updateProcessState('autosave', { status: 'Idle', detail: 'Next snapshot pending.' });
}

function tokenizeForEntropy(content) {
  const normalized = (content ?? '')
    .toLowerCase()
    .replace(/[“”]/g, '"')
    .replace(/[’]/g, "'");
  const tokens = normalized.match(/[a-z0-9]+/g);
  return tokens ? tokens.filter(Boolean) : [];
}

function buildEntropyVector(content) {
  return tokenizeForEntropy(content); // CODEx: Reuse lexical tokens for unified cosine similarity.
}

function computeEntropyScore(history, windowSize) {
  if (!Array.isArray(history) || history.length === 0) {
    return 0;
  }
  const volleyWindow = clampNumber(
    windowSize ?? config.entropyWindow ?? defaultConfig.entropyWindow,
    1,
    25,
    defaultConfig.entropyWindow
  );
  const sampleSize = Math.max(2, volleyWindow * 2);
  const relevant = history.filter(
    (entry) => entry && entry.countsAsTurn !== false && entry.speaker !== 'system' && typeof entry.content === 'string'
  );
  if (relevant.length < 2) {
    return 0;
  }
  const recent = relevant.slice(-sampleSize);
  if (recent.length < 2) {
    return 0;
  }
  const vectors = recent.map((entry) => buildEntropyVector(entry.content));
  let total = 0;
  let comparisons = 0;
  for (let index = 1; index < vectors.length; index += 1) {
    const similarity = cosineSimilarity(vectors[index - 1], vectors[index]);
    if (Number.isFinite(similarity)) {
      total += similarity;
      comparisons += 1;
    }
  }
  if (comparisons === 0) {
    return 0;
  }
  return total / comparisons;
}

function describeEntropy(score) {
  if (!Number.isFinite(score) || score <= 0) {
    return { state: 'calibrating', label: 'Calibrating…' };
  }
  if (score >= ENTROPY_LOOP_THRESHOLD) {
    return { state: 'loop', label: 'Looping' };
  }
  if (score <= ENTROPY_EXPLORE_THRESHOLD) {
    return { state: 'explore', label: 'Exploring' };
  }
  return { state: 'steady', label: 'Steady' };
}

function updateEntropyMeter() {
  if (!elements.entropyBarFill || !elements.entropyStatus) return;
  const score = computeEntropyScore(dualChatHistory, config.entropyWindow);
  currentEntropyScore = score;
  const percent = Math.max(0, Math.min(100, Math.round(score * 100)));
  elements.entropyBarFill.style.width = `${percent}%`;
  const description = describeEntropy(score);
  elements.entropyBarFill.dataset.state = description.state;
  const meter = elements.entropyBarFill.parentElement;
  if (meter) {
    meter.setAttribute('aria-valuenow', String(percent));
  }
  elements.entropyStatus.textContent = description.label;
  if (description.state !== lastEntropyState) {
    if (description.state === 'loop') {
      void maybeAdvancePhase();
    }
    lastEntropyState = description.state;
  }
}

function updateReflexStatus(message, state) {
  const defaultState = config.reflexEnabled ? 'ready' : 'disabled';
  const resolvedState = state ?? defaultState;
  const statusText = message
    ? message
    : config.reflexEnabled
      ? lastReflexSummaryAt
        ? `Last Reflex ${formatRelativeTime(lastReflexSummaryAt)} ago`
        : 'Reflex standing by'
      : 'Reflex disabled';
  if (elements.reflexStatus) {
    elements.reflexStatus.textContent = statusText;
    elements.reflexStatus.dataset.state = resolvedState;
  }
  if (elements.reflexStatusText) {
    elements.reflexStatusText.textContent = statusText;
    elements.reflexStatusText.dataset.state = resolvedState;
  }
}

async function maybeTriggerReflexSummary(speaker) {
  if (!config.reflexEnabled) return;
  if (reflexInFlight) return;
  if (!dualTurnsCompleted || config.reflexInterval <= 0) return;
  if (dualTurnsCompleted % config.reflexInterval !== 0) return;
  await runReflexSummary(speaker);
}

async function runReflexSummary(speaker, options = {}) {
  const { forced = false } = options;
  if (reflexInFlight) return;
  if (!forced && (!config.reflexEnabled || (!isDualChatRunning && !dualChatHistory.length))) {
    return;
  }
  if (!dualChatHistory.length) return;

  const agent = speaker === 'B' ? 'B' : 'A';
  const connection = activeDualConnections[agent] ?? getAgentConnection(agent);
  if (!connection.endpoint || !connection.model) {
    return;
  }
  reflexInFlight = true;
  updateReflexStatus('Reflex summarizing…', 'running');

  try {
    const partner = agent === 'A' ? 'B' : 'A';
    const agentName = getAgentDisplayName(agent);
    const partnerName = getAgentDisplayName(partner);
    const persona = getAgentPersona(agent);
    const volleyWindow = config.entropyWindow ?? defaultConfig.entropyWindow;
    const lookback = Math.max(volleyWindow * 2, config.reflexInterval ?? 4, 6);
    const recent = dualChatHistory.slice(-lookback);
    const transcript = recent
      .map((item) => {
        const tag = item.turnNumber ? `Turn ${item.turnNumber}` : 'Turn —';
        return `${tag} ${getAgentDisplayName(item.speaker)}: ${item.content}`;
      })
      .join('\n');
    const messages = [
      {
        role: 'system',
        content:
          (persona ? `${persona}\n\n` : '') +
          `${agentName}, you maintain a Reflex journal during debates with ${partnerName}. Capture fresh insights without repeating previous entries.`
      },
      {
        role: 'user',
        content:
          `Recent dialogue excerpt:\n${transcript}\n\nProduce a Reflex update with sections titled Insights, Questions, and Next moves. Each section should be one or two sentences. Include any novel vocabulary introduced.`
      }
    ];

    const { content: summary } = await callModel(messages, connection);
    if (!summary) {
      updateReflexStatus('Reflex ready (no summary returned)', 'ready');
      return;
    }
    const trimmedSummary = summary.trim();
    const topicFingerprints = extractTopicFingerprints(trimmedSummary);
    const fingerprintLine = topicFingerprints.length
      ? `\n\nTopic fingerprints: ${topicFingerprints.join(', ')}`
      : '';
    const decorated = `Reflex summary\n${trimmedSummary}${fingerprintLine}`;
    recordDualMessage(agent, decorated, {
      marker: 'reflex',
      label: 'Reflex',
      countsAsTurn: false,
      metadata: { topicFingerprints }
    });
    const reflexTimestamp = Date.now();
    lastReflexSummaryAt = reflexTimestamp;
    reflexSummaryCount += 1;
    void recordReflexHistory({
      timestamp: reflexTimestamp,
      speaker: agent,
      summary: trimmedSummary,
      keywords: topicFingerprints,
      entropy: currentEntropyScore,
      arenaTurn: dualTurnCounter,
      phaseId: activePhaseId ?? null,
      source: forced ? 'manual' : 'scheduled'
    });
    void recordLog('arena', `${agentName} posted Reflex summary #${reflexSummaryCount}`, { silent: true });
    archiveOldArenaTurns(WATCHDOG_ARCHIVE_PERCENT, { reason: 'reflex', silent: false }); // CODEx: Keep floating memory below pressure threshold.
    updateReflexStatus();
  } catch (error) {
    console.error('Reflex summary failed', error);
    const errorMessage = error.message || 'Unknown error occurred';
    addSystemMessage(`Reflex summary failed: ${errorMessage}`);
    updateReflexStatus('Reflex error (see console)', 'error');
    void recordLog('error', `Reflex summary failed: ${errorMessage}`, { level: 'error' });
  } finally {
    reflexInFlight = false;
  }
}

// CODEx: Persist Reflex metrics for later diagnostics.
async function recordReflexHistory(entry) {
  const payload = {
    timestamp: entry.timestamp ?? Date.now(),
    speaker: entry.speaker ?? 'system',
    summary: entry.summary ?? '',
    keywords: Array.isArray(entry.keywords) ? entry.keywords : [],
    entropy: Number.isFinite(entry.entropy) ? entry.entropy : currentEntropyScore,
    arenaTurn: entry.arenaTurn ?? dualTurnCounter,
    phaseId: entry.phaseId ?? activePhaseId ?? null,
    source: entry.source ?? 'auto'
  };
  try {
    await putReflexHistory(payload);
  } catch (error) {
    console.debug('Unable to store Reflex history record', error);
  }
}

function findLastTurnBySpeaker(agent) {
  for (let index = dualChatHistory.length - 1; index >= 0; index -= 1) {
    const entry = dualChatHistory[index];
    if (!entry) continue;
    if (entry.speaker === agent && entry.countsAsTurn !== false) {
      return entry;
    }
  }
  return null;
}

function extractReflexSegments(content) {
  const segments = { outcome: null, test: null };
  if (typeof content !== 'string') {
    return segments;
  }
  const pattern = /\[(Predicted Outcome|Test)[^\]]*?(?:→|->|=>|:)\s*([^\]\n\r]+)/gi;
  let match;
  while ((match = pattern.exec(content)) !== null) {
    const label = match[1]?.toLowerCase();
    const value = match[2]?.trim();
    if (!value) continue;
    if (label && label.includes('predicted') && !segments.outcome) {
      segments.outcome = value;
    } else if (label === 'test' && !segments.test) {
      segments.test = value;
    }
  }
  return segments;
}

function ensureSentence(text) {
  if (!text) return '';
  const trimmed = text.trim();
  if (!trimmed) return '';
  return /[.!?]$/.test(trimmed) ? trimmed : `${trimmed}.`;
}

function composeReflexSynthesisParagraph(segmentsA, segmentsB) {
  const aName = getAgentDisplayName('A');
  const bName = getAgentDisplayName('B');
  const sentences = [];
  if (segmentsA.outcome) {
    sentences.push(ensureSentence(`${aName} anticipates ${segmentsA.outcome}`));
  }
  if (segmentsB.outcome) {
    sentences.push(ensureSentence(`${bName} expects ${segmentsB.outcome}`));
  }
  const testClauses = [];
  if (segmentsA.test) {
    testClauses.push(`${aName} proposes ${segmentsA.test}`);
  }
  if (segmentsB.test) {
    testClauses.push(`${bName} suggests ${segmentsB.test}`);
  }
  if (testClauses.length) {
    sentences.push(ensureSentence(testClauses.join('; ')));
  }
  return sentences.join(' ');
}

function maybeTriggerReflexHooks() {
  if (!config.reflexEnabled) return;
  if (dualChatHistory.length < 2) return;
  const lastA = findLastTurnBySpeaker('A');
  const lastB = findLastTurnBySpeaker('B');
  if (!lastA || !lastB) return;
  const segmentsA = extractReflexSegments(lastA.content);
  const segmentsB = extractReflexSegments(lastB.content);
  if (!segmentsA.outcome || !segmentsA.test || !segmentsB.outcome || !segmentsB.test) {
    return;
  }
  const signature = `${normalizeTimestamp(lastA.timestamp)}|${normalizeTimestamp(lastB.timestamp)}`;
  if (lastReflexSynthesisSignature === signature) {
    return;
  }
  const summaryParagraph = composeReflexSynthesisParagraph(segmentsA, segmentsB);
  if (!summaryParagraph) {
    return;
  }
  const topicFingerprints = extractTopicFingerprints(summaryParagraph);
  const fingerprintLine = topicFingerprints.length
    ? `\n\nTopic fingerprints: ${topicFingerprints.join(', ')}`
    : '';
  const content = `Reflex summary\n${summaryParagraph}${fingerprintLine}`;
  recordDualMessage('system', content, {
    marker: 'reflex',
    label: 'Reflex',
    countsAsTurn: false,
    metadata: { topicFingerprints }
  });
  const reflexTimestamp = Date.now();
  lastReflexSummaryAt = reflexTimestamp;
  reflexSummaryCount += 1;
  lastReflexSynthesisSignature = signature;
  updateReflexStatus();
  void recordReflexHistory({
    timestamp: reflexTimestamp,
    speaker: 'system',
    summary: summaryParagraph,
    keywords: topicFingerprints,
    entropy: currentEntropyScore,
    arenaTurn: dualTurnCounter,
    phaseId: activePhaseId ?? null,
    source: 'auto-hook'
  });
  void recordLog('arena', 'Reflex auto-synthesis posted.', { silent: true });
}

async function readPhasePromptFile(fileName) {
  if (!fileName) return null;
  if (phasePromptCache.has(fileName)) {
    return phasePromptCache.get(fileName);
  }
  const relativePath = `${PHASE_PROMPT_DIR}${fileName}`;
  const fsResult = readStandaloneFile(relativePath);
  if (fsResult?.content) {
    phasePromptCache.set(fileName, fsResult.content);
    return fsResult.content;
  }
  try {
    const response = await fetch(relativePath, { cache: 'no-store' });
    if (response.ok) {
      const text = await response.text();
      phasePromptCache.set(fileName, text);
      return text;
    }
  } catch (error) {
    console.warn('Failed to fetch phase prompt', relativePath, error);
  }
  return null;
}

async function ensurePhasePromptBaseline() { // CODEx: Guarantee the primary phase prompt exists for arena rotation.
  const baselinePath = `${PHASE_PROMPT_DIR}phase1.txt`; // CODEx: Target baseline prompt path.
  const directory = PHASE_PROMPT_DIR.replace(/\/$/, ''); // CODEx: Normalize directory path without trailing slash.
  const existing = readStandaloneFile(baselinePath); // CODEx: Attempt to read from standalone bridge first.
  if (existing?.content) {
    return true; // CODEx: Baseline already provisioned.
  }
  let seeded = false; // CODEx: Track whether a file was created during this check.
  if (ensureStandaloneDirectory(directory)) { // CODEx: Ensure directory exists when bridge available.
    const defaultContent = [
      '# Flow Dynamics Primer', // CODEx: Provide concise placeholder guidance for phase one.
      'Use this phase to establish shared definitions, goals, and system constraints before diving deeper.', // CODEx: Explain the intent of the initial debate phase.
      'Highlight any conflicting assumptions so later phases can resolve them efficiently.' // CODEx: Encourage surfacing disagreements early.
    ].join('\n');
    seeded = writeStandaloneFile(baselinePath, defaultContent, { contentType: 'text/plain' }) !== false; // CODEx: Seed file via bridge when possible.
    if (seeded) {
      console.log(`[RAG] Seeded baseline phase prompt at ${baselinePath}`); // CODEx: Log successful creation for diagnostics.
      phasePromptCache.set('phase1.txt', defaultContent); // CODEx: Prime cache with seeded content.
      return true; // CODEx: Early exit after successful creation.
    }
  }
  try {
    const response = await fetch(baselinePath, { cache: 'no-store' }); // CODEx: Probe bundled asset when filesystem write unavailable.
    if (response.ok) {
      return true; // CODEx: Asset already present on the server.
    }
  } catch (error) {
    console.debug('[RAG] Phase prompt fetch fallback failed', error); // CODEx: Emit debug trace without interrupting startup.
  }
  if (!seeded) {
    console.warn(`[RAG] Baseline phase prompt missing at ${baselinePath}.`); // CODEx: Surface absence so operators can provision it.
  }
  return false; // CODEx: Signal that prompt could not be confirmed.
}

async function activatePhase(index, options = {}) {
  if (!Array.isArray(phases) || phases.length === 0) return;
  if (index < 0 || index >= phases.length) return;
  const phase = phases[index];
  const alreadyActive = activePhaseId === phase.id && activePhaseIndex === index;
  if (alreadyActive && !options.force) {
    return;
  }
  let content = null;
  if (phase.prompt) {
    content = await readPhasePromptFile(phase.prompt);
  }
  activePhaseIndex = index;
  activePhaseId = phase.id;
  pendingPhaseAdvance = false;
  lastEntropyState = 'calibrating';
  lastReflexSynthesisSignature = null;
  if (!content) {
    const message = `Phase ${phase.id} (${phase.title}) prompt not found at ${PHASE_PROMPT_DIR}${phase.prompt || ''}.`;
    addSystemMessage(message);
    void recordLog('error', message, { level: 'warn' });
    return;
  }
  const trimmed = content.trim();
  const header = `Phase ${phase.id}: ${phase.title}`;
  const body = trimmed ? `${header}\n${trimmed}` : header;
  recordDualMessage('system', body, {
    marker: 'phase',
    label: `Phase ${phase.id}`,
    countsAsTurn: false
  });
  if (options.announce) {
    addSystemMessage(`Loaded ${header} from ${PHASE_PROMPT_DIR}.`);
  }
  void recordLog('arena', `Activated phase ${phase.id} – ${phase.title}`, { silent: true });
}

async function maybeAdvancePhase() {
  if (!isDualChatRunning) return;
  if (pendingPhaseAdvance) return;
  if (!Array.isArray(phases) || phases.length === 0) return;
  if (activePhaseIndex >= phases.length - 1) return;
  pendingPhaseAdvance = true;
  try {
    await activatePhase(activePhaseIndex + 1, { auto: true });
  } finally {
    pendingPhaseAdvance = false;
  }
}

function createDualStreamHandle(speaker, speakerName) {
  if (!elements.dualMessageTemplate || !elements.dualChatWindow) {
    return null; // CODEx: Skip placeholder when template is unavailable.
  }
  const node = elements.dualMessageTemplate.content.firstElementChild.cloneNode(true); // CODEx: Clone dual message template.
  node.classList.add('dual-message--streaming'); // CODEx: Flag streaming placeholder.
  node.dataset.streaming = 'true'; // CODEx: Surface state for styling/debugging.
  const timestamp = Date.now(); // CODEx: Track placeholder creation time.
  const speakerEl = node.querySelector('.dual-speaker');
  if (speakerEl) {
    speakerEl.textContent = speakerName; // CODEx: Render speaker name immediately.
  }
  const turnEl = node.querySelector('.dual-turn');
  if (turnEl) {
    turnEl.textContent = '';
    turnEl.hidden = true; // CODEx: Hide turn index until final commit.
  }
  const badgeEl = node.querySelector('.dual-badge');
  if (badgeEl) {
    badgeEl.textContent = 'streaming…'; // CODEx: Indicate active stream state.
    badgeEl.hidden = false;
  }
  const timeEl = node.querySelector('.dual-time');
  if (timeEl) {
    timeEl.textContent = formatTimestamp(timestamp); // CODEx: Timestamp placeholder for continuity.
  }
  const contentEl = node.querySelector('.dual-content');
  if (contentEl) {
    contentEl.textContent = '…'; // CODEx: Seed placeholder text.
  }
  elements.dualChatWindow.appendChild(node); // CODEx: Attach streaming placeholder to arena view.
  scrollContainerToBottom(elements.dualChatWindow); // CODEx: Ensure placeholder is visible.
  return { node, contentEl, badgeEl, timestamp, speaker }; // CODEx: Return handle for incremental updates.
}

function updateDualStreamHandle(handle, text) {
  if (!handle || !handle.node) {
    return; // CODEx: Guard against absent handles.
  }
  const payload = text && text.trim() ? text : '…';
  if (handle.contentEl) {
    handle.contentEl.textContent = payload; // CODEx: Update placeholder content text.
  }
  if (handle.badgeEl) {
    handle.badgeEl.textContent = payload === '…' ? 'streaming…' : 'live'; // CODEx: Reflect stream progress.
    handle.badgeEl.hidden = false;
  }
}

function finalizeDualStreamHandle(handle) {
  if (!handle || !handle.node) {
    return; // CODEx: Nothing to finalize.
  }
  handle.node.classList.remove('dual-message--streaming'); // CODEx: Remove streaming styling.
  handle.node.dataset.streaming = 'false'; // CODEx: Reset state flag.
  if (handle.badgeEl) {
    handle.badgeEl.hidden = true; // CODEx: Hide streaming badge once finalized.
  }
}

async function generateDualTurn(speaker, seed, options = {}) {
  if (!isDualChatRunning) return;
  const partner = speaker === 'A' ? 'B' : 'A';
  const speakerName = getAgentDisplayName(speaker);
  const partnerName = getAgentDisplayName(partner);
  const normalizedSeed = typeof seed === 'string' && seed.trim() ? seed : activeDualSeed || config.dualSeed || DEFAULT_DUAL_SEED;
  const persona = getAgentPersona(speaker);
  const connection = activeDualConnections[speaker] ?? getAgentConnection(speaker);
  const reasoningActive = isReasoningModeActive({
    model: connection.model,
    providerPreset: connection.providerPreset
  }); // CODEx: Detect reasoning mode per agent connection.
  clearDualCountdownTimer();
  if (elements.dualStatus) {
    elements.dualStatus.textContent = `${speakerName} is thinking…`;
  }
  if (!connection.endpoint || !connection.model) {
    addSystemMessage(`${speakerName} lost its connection details. Stop and reconfigure before resuming the arena.`);
    stopDualChat();
    return;
  }

  const processKey = speaker === 'A' ? 'modelA' : 'modelB';
  const autoInjecting = Boolean(config.autoInjectMemories);
  updateProcessState(processKey, {
    status: 'Retrieving',
    detail: autoInjecting
      ? `${speakerName} is gathering arena memories.`
      : `${speakerName} is skipping auto retrieval.`
  });

  const historyMessages = [];
  historyMessages.push({
    role: 'system',
    content: persona || `You are ${speakerName}, an autonomous AI who is collaborating with ${partnerName}. Reply as ${speakerName}.`
  });

  const historyLookback = reasoningActive ? Math.min(dualChatHistory.length, 6) : dualChatHistory.length; // CODEx: Restrict history depth for reasoning safety.
  const historySource = dualChatHistory.slice(-historyLookback); // CODEx: Extract relevant history window.

  for (const turn of historySource) {
    if (!turn) continue;
    if (turn.speaker === 'system' || turn.marker === 'reflex' || turn.marker === 'phase') {
      historyMessages.push({ role: 'system', content: turn.content });
      continue;
    }
    const label = turn.speaker === speaker ? speakerName : partnerName;
    const role = turn.speaker === speaker ? 'assistant' : 'user';
    historyMessages.push({ role, content: `${label}: ${turn.content}` });
  }

  const latestPartnerTurn = dualChatHistory.filter((turn) => turn.speaker === partner).slice(-1)[0];
  if (options.isInitial) {
    historyMessages.push({
      role: 'user',
      content: `${partnerName}: ${normalizedSeed}`
    });
  } else {
    const lastMessage = historyMessages[historyMessages.length - 1];
    if (!lastMessage || lastMessage.role !== 'user') {
      const fallbackContent = latestPartnerTurn
        ? `${partnerName}: ${latestPartnerTurn.content}`
        : `${partnerName}: ${normalizedSeed}`;
      historyMessages.push({ role: 'user', content: fallbackContent });
    }
  }

  const retrievalQuery = options.isInitial ? seed : latestPartnerTurn?.content ?? seed;
  let retrievedMemories = [];
  if (autoInjecting && retrievalQuery) {
    const retrievalCap = reasoningActive
      ? Math.max(0, Math.min(2, config.retrievalCount || 2))
      : config.retrievalCount; // CODEx: Lighten RAG load under reasoning pressure.
    if (retrievalCap !== 0) {
      retrievedMemories = await retrieveRelevantMemories(retrievalQuery, retrievalCap, { speaker });
    }
    if (reasoningActive && retrievedMemories.length > 2) {
      retrievedMemories = retrievedMemories.slice(0, 2); // CODEx: Cap inserted cues when reasoning.
    }
    if (retrievedMemories.length) {
      const compiled = retrievedMemories
        .map((item) => `• ${formatTimestamp(item.timestamp)} ${formatMemoryPreview(item.content, reasoningActive ? 140 : 220)}`)
        .join('\n'); // CODEx: Compress cue payload for reasoning focus.
      historyMessages.push({
        role: 'system',
        content: reasoningActive
          ? `Brief memory cues:\n${compiled}`
          : `Shared long-term memories:\n${compiled}` // CODEx: Swap to light cue format when reasoning.
      });
    }
  }
  if (autoInjecting) {
    registerRetrieval(speaker, retrievedMemories.length);
  }

  const streamHandle = createDualStreamHandle(speaker, speakerName); // CODEx: Prime arena UI for streaming output.

  try {
    updateProcessState(processKey, {
      status: 'Contacting model',
      detail: `Calling ${connection.model} at ${connection.endpoint}`
    });
    if (speaker === 'A') {
      modelAInFlight = true;
    } else {
      modelBInFlight = true;
    }
    const { content: reply, truncated, reasoning } = await callModel(historyMessages, {
      ...connection,
      onStream: (payload) => {
        if (!streamHandle) return; // CODEx: Skip when placeholder unavailable.
        if (payload?.type === 'content') {
          updateDualStreamHandle(streamHandle, payload.text ?? ''); // CODEx: Render incremental text.
        }
      }
    });
    if (!reply) {
      addSystemMessage(`${speakerName} did not respond.`);
      stopDualChat();
      return;
    }
    if (truncated) {
      addSystemMessage(
        `${speakerName} reply trimmed to stay within the ~${config.maxResponseTokens} token response guard.`
      );
    }
    finalizeDualStreamHandle(streamHandle); // CODEx: Remove streaming badge before committing history.
    const badgeParts = [];
    if (retrievedMemories.length) {
      badgeParts.push(`${retrievedMemories.length} memories`);
    }
    if (reasoning) {
      badgeParts.push('reasoning');
    }
    if (truncated) {
      badgeParts.push('trimmed');
    }
    recordDualMessage(speaker, reply, {
      marker: truncated ? 'truncated' : undefined,
      label: badgeParts.length ? badgeParts.join(' • ') : undefined,
      existingNode: streamHandle?.node,
      contentNode: streamHandle?.contentEl,
      timestamp: streamHandle?.timestamp
    });
    updateProcessState(processKey, {
      status: 'Reply delivered',
      detail: `${speakerName} responded with ~${estimateTokenCount(reply)} tokens.`
    });
    nextDualSpeaker = partner;
    dualTurnsCompleted += 1;
    void maybeTriggerReflexSummary(speaker);
    updateEntropyMeter();
  } catch (error) {
    if (streamHandle?.node?.isConnected) {
      streamHandle.node.remove(); // CODEx: Clear placeholder on failure.
    }
    const errorMessage = error.message || 'Unknown error occurred';
    addSystemMessage(`Dual chat error: ${errorMessage}`);
    updateProcessState(processKey, { status: 'Error', detail: errorMessage });
    void recordLog('error', `Dual chat error for ${speakerName}: ${errorMessage}`, { level: 'error' });
    stopDualChat();
  } finally {
    if (speaker === 'A') {
      modelAInFlight = false;
    } else {
      modelBInFlight = false;
    }
    setTimeout(() => {
      const stillRunning = isDualChatRunning;
      const disabled = speaker === 'B' && !config.agentBEnabled;
      const detail = stillRunning
        ? `${speakerName} awaiting next arena turn.`
        : disabled
          ? 'Model B is disabled.'
          : 'Awaiting arena start.';
      const status = stillRunning ? 'Idle' : disabled ? 'Disabled' : 'Idle';
      updateProcessState(processKey, { status, detail });
    }, 500);
  }
}

function recordDualMessage(speaker, content, options = {}) {
  const {
    saveHistory = true,
    label,
    marker,
    turnNumber: providedTurn,
    existingNode,
    contentNode,
    timestamp: providedTimestamp
  } = options; // CODEx: Support streaming placeholders when recording turns.
  const metadata = sanitizeMetadata(options.metadata);
  const countsOption = options.countsAsTurn;
  const timestamp = Number.isFinite(providedTimestamp) ? providedTimestamp : Date.now(); // CODEx: Preserve streaming timestamps.
  let turnNumber = typeof providedTurn === 'number' ? providedTurn : null;
  const normalizedSpeaker = speaker === 'B' ? 'B' : speaker === 'system' ? 'system' : 'A';
  const countsAsTurn = typeof countsOption === 'boolean' ? countsOption : normalizedSpeaker !== 'system';

  const speakerName = getAgentDisplayName(normalizedSpeaker);

  if (saveHistory) {
    if (countsAsTurn) {
      if (normalizedSpeaker === 'A') {
        dualTurnCounter += 1;
        if (turnNumber === null) {
          turnNumber = dualTurnCounter;
        }
      } else if (normalizedSpeaker === 'B') {
        if (dualTurnCounter === 0) {
          dualTurnCounter = 1;
        }
        if (turnNumber === null) {
          turnNumber = dualTurnCounter;
        }
      }
    } else if (turnNumber === null && dualChatHistory.length) {
      turnNumber = dualChatHistory[dualChatHistory.length - 1].turnNumber ?? dualTurnCounter;
    }
    dualChatHistory.push({
      speaker: normalizedSpeaker,
      content,
      timestamp,
      turnNumber,
      marker,
      countsAsTurn,
      metadata
    });

    const memoryEntry = {
      id: `arena-${timestamp}-${normalizedSpeaker}-${dualChatHistory.length}`,
      role: speakerName,
      speaker: normalizedSpeaker,
      speakerName,
      content,
      timestamp,
      origin: 'arena',
      turnNumber,
      mode: MODE_ARENA,
      marker,
      countsAsTurn,
      phaseId: activePhaseId ?? null,
      metadata
    };
    floatingMemory.push(memoryEntry);
    trimFloatingMemory();
    queueMemoryCheckpoint(MODE_ARENA, memoryEntry);
    void persistMessage(memoryEntry);
  }

  const templateNode = elements.dualMessageTemplate?.content?.firstElementChild;
  const node = existingNode ?? (templateNode ? templateNode.cloneNode(true) : null);
  if (!node) {
    return; // CODEx: Bail when template is missing and no placeholder provided.
  }
  if (existingNode) {
    node.classList.remove('dual-message--streaming'); // CODEx: Ensure finalized styling.
    node.dataset.streaming = 'false';
  }
  const turnNode = node.querySelector('.dual-turn');
  if (turnNode) {
    if (turnNumber) {
      turnNode.textContent = `Turn ${turnNumber}`;
      turnNode.hidden = false;
    } else {
      turnNode.textContent = '';
      turnNode.hidden = true;
    }
  }
  const badgeNode = node.querySelector('.dual-badge');
  if (badgeNode) {
    if (label) {
      badgeNode.textContent = label;
      badgeNode.hidden = false;
    } else {
      badgeNode.textContent = '';
      badgeNode.hidden = true;
    }
  }
  node.querySelector('.dual-speaker').textContent = speakerName;
  const timeNode = node.querySelector('.dual-time');
  if (timeNode) {
    timeNode.textContent = formatTimestamp(timestamp); // CODEx: Use supplied timestamp for continuity.
  }
  const contentTarget = contentNode ?? node.querySelector('.dual-content');
  if (contentTarget) {
    contentTarget.textContent = content; // CODEx: Finalize streamed content.
  }
  if (marker) {
    node.classList.add(`dual-message--${marker}`);
  }
  if (!existingNode) {
    elements.dualChatWindow.appendChild(node); // CODEx: Append fresh entry when no placeholder exists.
    scrollContainerToBottom(elements.dualChatWindow);
  }
  void persistDualRagSnapshot();
  if (countsAsTurn && (normalizedSpeaker === 'A' || normalizedSpeaker === 'B')) {
    maybeTriggerReflexHooks();
  }
  updateEntropyMeter();
}

function exportDualTranscript() {
  if (!dualChatHistory.length) {
    addSystemMessage('No arena conversation is available to export yet.');
    return;
  }

  void persistDualRagSnapshot();

  const exportTimestamp = new Date();
  const timestampSlug = exportTimestamp.toISOString().replace(/[:.]/g, '-');
  const seed = activeDualSeed || config.dualSeed || DEFAULT_DUAL_SEED;
  const transcriptLines = [
    'SAM Arena transcript',
    `Exported at: ${exportTimestamp.toLocaleString()}`,
    `Conversation seed: ${seed}`,
    ''
  ];

  dualChatHistory.forEach((item, index) => {
    const turn = item.turnNumber ?? index + 1;
    const speakerName = getAgentDisplayName(item.speaker);
    const time = new Date(normalizeTimestamp(item.timestamp)).toLocaleString();
    transcriptLines.push(`Turn ${turn} • ${speakerName} @ ${time}`);
    transcriptLines.push(item.content);
    transcriptLines.push('');
  });

  const transcript = transcriptLines.join('\n');
  const blob = new Blob([transcript], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `sam-arena-${timestampSlug}.txt`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  setTimeout(() => URL.revokeObjectURL(url), 0);

  addSystemMessage('Arena transcript exported.');
}
