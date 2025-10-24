const MB = 1024 * 1024;
const MIN_MEMORY_LIMIT_MB = 50;
const MAX_MEMORY_LIMIT_MB = 200;
const MEMORY_LIMIT_STEP_MB = 10;

const DEFAULT_DUAL_SEED = 'Debate whether persistent memories make AI more helpful.';

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
  autoInjectMemories: false,
  requestTimeoutSeconds: 30,
  reasoningTimeoutSeconds: 120,
  contextTurns: 12,
  endpoint: 'http://localhost:1234/v1/chat/completions',
  model: 'lmstudio-community/Meta-Llama-3-8B-Instruct',
  apiKey: '',
  systemPrompt: 'You are SAM, a helpful memory-augmented assistant who reflects on long-term memories when they are relevant.',
  temperature: 0.7,
  maxResponseTokens: 512,
  autoSpeak: false,
  ttsPreset: 'browser',
  ttsServerUrl: '',
  ttsVoiceId: '',
  ttsApiKey: '',
  ttsVolume: 100,
  agentAName: 'SAM-A',
  agentAPrompt: 'You are SAM-A, an analytical AI researcher who values precision and structure.',
  agentBEnabled: false,
  agentBName: 'SAM-B',
  agentBPrompt: 'You are SAM-B, a creative explorer who loves bold ideas and storytelling.',
  agentBProviderPreset: 'inherit',
  agentBEndpoint: '',
  agentBModel: '',
  agentBApiKey: '',
  dualAutoContinue: true,
  dualTurnLimit: 0,
  dualTurnDelaySeconds: 30,
  dualSeed: DEFAULT_DUAL_SEED,
  reflexEnabled: true,
  reflexInterval: 8,
  entropyWindow: 6,
  backgroundSource: 'default',
  backgroundImage: '',
  backgroundUrl: '',
  debugEnabled: false
};

const TEST_TTS_PHRASE = 'This is SAM performing a voice check with persistent memory engaged.';
const TTS_ERROR_COOLDOWN = 15000;
const PINNED_STORAGE_KEY = 'sam-pinned-messages';
const RAM_DISK_CHUNK_PREFIX = 'sam-chunked-';
const RAM_DISK_UNCHUNKED_PREFIX = 'sam-unchunked-';
const RAM_DISK_ARCHIVE_PREFIX = 'sam-archive-';
const RAM_DISK_MANIFEST_PREFIX = 'sam-manifest-';
const REASONING_MODEL_HINTS = [
  'reason',
  'cogito',
  'sonar',
  'think',
  'deepseek-r1',
  'deepseek_reasoner',
  'reasoner',
  'o1',
  'o3'
];

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
    endpoint: 'http://localhost:11434/v1/chat/completions',
    model: 'tinyllama',
    requiresKey: false,
    contextLimit: 65536
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
  loaded: false
};
const ramDiskCache = {
  chunked: new Map(),
  unchunked: new Map(),
  archives: new Map(),
  manifests: new Map()
};

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
  agentBEnabled: document.getElementById('agentBEnabled'),
  agentBConfig: document.getElementById('agentBConfig'),
  messageInput: document.getElementById('messageInput'),
  sendButton: document.getElementById('sendButton'),
  voiceButton: document.getElementById('voiceButton'),
  speakToggle: document.getElementById('speakToggle'),
  retrieveButton: document.getElementById('retrieveButton'),
  clearChatButton: document.getElementById('clearChatButton'),
  memorySlider: document.getElementById('memorySlider'),
  memorySliderValue: document.getElementById('memorySliderValue'),
  retrievalCount: document.getElementById('retrievalCount'),
  autoInjectMemories: document.getElementById('autoInjectMemories'),
  contextTurns: document.getElementById('contextTurns'),
  exportButton: document.getElementById('exportMemoryButton'),
  floatingMemoryList: document.getElementById('floatingMemoryList'),
  floatingMemoryCount: document.getElementById('floatingMemoryCount'),
  pinnedMemoryCount: document.getElementById('pinnedMemoryCount'),
  ragMemoryCount: document.getElementById('ragMemoryCount'),
  ragSnapshotCount: document.getElementById('ragSnapshotCount'),
  ragFootprint: document.getElementById('ragFootprint'),
  ragImportStatus: document.getElementById('ragImportStatus'),
  refreshFloatingButton: document.getElementById('refreshFloatingButton'),
  loadRagButton: document.getElementById('loadRagButton'),
  loadChunkDataButton: document.getElementById('loadChunkDataButton'),
  saveChatArchiveButton: document.getElementById('saveChatArchiveButton'),
  archiveUnpinnedButton: document.getElementById('archiveUnpinnedButton'),
  modelARetrievals: document.getElementById('modelARetrievals'),
  modelBRetrievals: document.getElementById('modelBRetrievals'),
  endpointInput: document.getElementById('endpointInput'),
  providerSelect: document.getElementById('providerSelect'),
  providerNotes: document.getElementById('providerNotes'),
  modelInput: document.getElementById('modelInput'),
  apiKeyInput: document.getElementById('apiKeyInput'),
  systemPromptInput: document.getElementById('systemPromptInput'),
  temperatureInput: document.getElementById('temperatureInput'),
  maxTokensInput: document.getElementById('maxTokensInput'),
  maxTokensHint: document.getElementById('maxTokensHint'),
  saveConfigButton: document.getElementById('saveConfigButton'),
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
  agentAPrompt: document.getElementById('agentAPrompt'),
  agentBName: document.getElementById('agentBName'),
  agentBPrompt: document.getElementById('agentBPrompt'),
  arenaPrimaryConfig: document.getElementById('arenaPrimaryConfig'),
  requestTimeout: document.getElementById('requestTimeout'),
  reasoningTimeout: document.getElementById('reasoningTimeout'),
  agentBProviderSelect: document.getElementById('agentBProviderSelect'),
  agentBProviderNotes: document.getElementById('agentBProviderNotes'),
  agentBEndpoint: document.getElementById('agentBEndpoint'),
  agentBModel: document.getElementById('agentBModel'),
  agentBApiKey: document.getElementById('agentBApiKey'),
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
  primeRamDiskCache();
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
    config.maxResponseTokens = Math.round(clampedMaxTokens);
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
      2,
      50,
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
    config.autoInjectMemories = Boolean(config.autoInjectMemories);
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
  elements.providerNotes.textContent = parts.filter(Boolean).join(' ');
}

function applyArenaProviderPreset(agent, presetId, { silent = false } = {}) {
  if (agent === 'A') {
    return;
  }

  const prefix = 'agentB';
  if (!presetId || presetId === 'inherit') {
    config[`${prefix}ProviderPreset`] = 'inherit';
    updateAgentConnectionInputs(agent);
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
  saveConfig();
  updateModelConnectionStatus();
  if (!silent && preset.id !== 'custom') {
    addSystemMessage(
      `${getAgentDisplayName(agent)} loaded ${preset.label} defaults. Provide credentials if required and click "Save model settings".`
    );
  }
}

function getAgentDisplayName(agent) {
  return agent === 'A' ? config.agentAName || 'SAM-A' : config.agentBName || 'SAM-B';
}

function getAgentPersona(agent) {
  const raw = agent === 'A' ? config.agentAPrompt : config.agentBPrompt;
  return raw?.trim() ?? '';
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
  elements.contextTurns.value = config.contextTurns;
  if (elements.providerSelect) {
    elements.providerSelect.value = config.providerPreset ?? defaultConfig.providerPreset;
  }
  elements.endpointInput.value = config.endpoint;
  elements.modelInput.value = config.model;
  elements.apiKeyInput.value = config.apiKey;
  if (elements.requestTimeout) {
    elements.requestTimeout.value = config.requestTimeoutSeconds;
  }
  if (elements.reasoningTimeout) {
    elements.reasoningTimeout.value = config.reasoningTimeoutSeconds;
  }
  elements.systemPromptInput.value = config.systemPrompt;
  elements.temperatureInput.value = config.temperature;
  if (elements.maxTokensInput) {
    updateMaxTokensCeiling();
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
  elements.agentAPrompt.value = config.agentAPrompt;
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

  updateArenaProviderNotes(agent, connection);
}

function applyAgentBEnabledState() {
  const enabled = config.agentBEnabled !== false;
  if (elements.agentBEnabled) {
    elements.agentBEnabled.checked = enabled;
  }
  if (elements.arenaPrimaryConfig) {
    elements.arenaPrimaryConfig.hidden = !enabled;
    elements.arenaPrimaryConfig.setAttribute('aria-hidden', String(!enabled));
  }
  if (elements.agentBConfig) {
    elements.agentBConfig.hidden = !enabled;
    elements.agentBConfig.setAttribute('aria-hidden', String(!enabled));
  }
  const toggledFields = [
    'agentAPrompt',
    'agentBProviderSelect',
    'agentBEndpoint',
    'agentBModel',
    'agentBApiKey',
    'agentBName',
    'agentBPrompt'
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
  if (!enabled && elements.agentBProviderNotes) {
    elements.agentBProviderNotes.textContent = 'Enable Model B to configure a separate connection.';
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
      temperature: config.temperature,
      maxTokens: config.maxResponseTokens,
      providerPreset: basePreset?.id ?? 'custom',
      providerLabel: basePreset ? `${basePreset.label} (main)` : 'Custom',
      presetDescription: basePreset?.description ?? '',
      requiresKey: Boolean(basePreset?.requiresKey),
      inherits: false,
      agentPresetId: basePreset?.id ?? 'custom'
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

  return {
    endpoint,
    model,
    apiKey,
    temperature: config.temperature,
    maxTokens: config.maxResponseTokens,
    providerPreset: preset?.id ?? 'custom',
    providerLabel: inherits && basePreset ? `${basePreset.label} (main)` : preset?.label ?? 'Custom',
    presetDescription: preset?.description ?? '',
    requiresKey: Boolean(preset?.requiresKey),
    inherits,
    agentPresetId
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

  addListener(elements.autoInjectMemories, 'change', (event) => {
    config.autoInjectMemories = event.target.checked;
    saveConfig();
    updateRagStatusDisplay();
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
  });

  addListener(elements.modelInput, 'input', (event) => {
    config.model = event.target.value.trim();
    updateAgentConnectionInputs('B');
  });

  addListener(elements.apiKeyInput, 'input', (event) => {
    config.apiKey = event.target.value;
    updateAgentConnectionInputs('B');
  });

  addListener(elements.requestTimeout, 'change', (event) => {
    config.requestTimeoutSeconds = clampNumber(event.target.value, 5, 600, config.requestTimeoutSeconds);
    elements.requestTimeout.value = config.requestTimeoutSeconds;
    if (config.reasoningTimeoutSeconds < config.requestTimeoutSeconds) {
      config.reasoningTimeoutSeconds = config.requestTimeoutSeconds;
      if (elements.reasoningTimeout) {
        elements.reasoningTimeout.value = config.reasoningTimeoutSeconds;
      }
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
    }
    saveConfig();
  });

  addListener(elements.reasoningTimeout, 'change', (event) => {
    config.reasoningTimeoutSeconds = clampNumber(event.target.value, 15, 900, config.reasoningTimeoutSeconds);
    if (config.reasoningTimeoutSeconds < config.requestTimeoutSeconds) {
      config.reasoningTimeoutSeconds = config.requestTimeoutSeconds;
    }
    elements.reasoningTimeout.value = config.reasoningTimeoutSeconds;
    saveConfig();
  });

  addListener(elements.reasoningTimeout, 'input', (event) => {
    config.reasoningTimeoutSeconds = clampNumber(event.target.value, 15, 900, config.reasoningTimeoutSeconds);
    if (config.reasoningTimeoutSeconds < config.requestTimeoutSeconds) {
      config.reasoningTimeoutSeconds = config.requestTimeoutSeconds;
    }
    elements.reasoningTimeout.value = config.reasoningTimeoutSeconds;
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

  addListener(elements.temperatureInput, 'change', (event) => {
    const parsed = Number.parseFloat(event.target.value);
    if (!Number.isNaN(parsed)) {
      config.temperature = clampNumber(parsed, 0, 2, config.temperature);
    }
    elements.temperatureInput.value = config.temperature.toString();
  });

  addListener(elements.maxTokensInput, 'change', (event) => {
    const parsed = Number.parseInt(event.target.value, 10);
    const limit = getProviderContextLimit(config.providerPreset);
    if (Number.isNaN(parsed) || parsed <= 0) {
      config.maxResponseTokens = Math.round(
        clampNumber(defaultConfig.maxResponseTokens, 16, limit, defaultConfig.maxResponseTokens)
      );
    } else {
      config.maxResponseTokens = Math.round(
        clampNumber(parsed, 16, limit, config.maxResponseTokens ?? defaultConfig.maxResponseTokens)
      );
    }
    updateMaxTokensCeiling();
    if (activeDualConnections.A) {
      activeDualConnections.A.maxTokens = config.maxResponseTokens;
    }
    if (activeDualConnections.B) {
      activeDualConnections.B.maxTokens = config.maxResponseTokens;
    }
    saveConfig();
  });

  addListener(elements.saveConfigButton, 'click', () => {
    saveConfig();
    updateModelConnectionStatus();
    if (document.body.classList.contains('options-open')) {
      closeOptionsPanel();
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
    config.entropyWindow = clampNumber(value, 2, 50, defaultConfig.entropyWindow);
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

  addListener(elements.agentAPrompt, 'input', (event) => {
    config.agentAPrompt = event.target.value;
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
  const current = config.maxResponseTokens ?? defaultConfig.maxResponseTokens;
  const normalized = Math.round(clampNumber(current, 16, limit, current));
  config.maxResponseTokens = normalized;
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
      const messages = chunkList.map((chunk, index) => ({
        id: `${originalPath}-chunk-${index}`,
        role: 'archive',
        content: chunk.content,
        timestamp: Date.now() + index,
        origin: 'rag-chunked',
        mode: MODE_CHAT,
        pinned: false
      }));

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
    try {
      const response = await fetch(path);
      if (response.ok) {
        const payload = await response.json();
        const normalized = { ...payload, chunkedPath: path };
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
        const chunks = await chunkFile(file, chunkerState.chunkSize, chunkerState.chunkOverlap);
        if (Array.isArray(chunks) && chunks.length > 0) {
          const chunkedData = {
            originalPath,
            chunkSize: chunkerState.chunkSize,
            chunkOverlap: chunkerState.chunkOverlap,
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
          startIndex: chunks.length * (chunkSize - overlap)
        });

        const overlapText = extractOverlapText(currentChunk, overlap);
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
        startIndex: chunks.length * (chunkSize - overlap)
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

async function saveChunkedFileToFS(path, data) {
  const payloadForCache = { ...data, chunkedPath: path };
  cacheChunkedInRam([path, data?.originalPath], payloadForCache);
  try {
    const response = await fetch(path, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data, null, 2)
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
        deletePromises.push(
          fetch(file.chunkedPath, { method: 'DELETE' })
            .then(() => addSystemMessage(`✓ Deleted ${file.chunkedPath}`))
            .catch(() => addSystemMessage(`⚠️ Could not delete ${file.chunkedPath}`))
        );
      }
    });
    chunkerState.unchunkedFiles.forEach(file => {
      if (file.path && file.path.includes('/unchunked/')) {
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

  if (shouldAutoRetrieve) {
    retrievedMemories = await retrieveRelevantMemories(userEntry.content, config.retrievalCount);
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

  const recent = getRecentConversation();
  for (const item of recent) {
    messages.push({ role: item.role, content: item.content });
  }

  messages.push({ role: 'user', content: userEntry.content });
  return { messages, retrievedMemories };
}

function getRecentConversation() {
  const limit = Math.max(2, Math.min(config.contextTurns, conversationLog.length));
  return conversationLog.slice(-limit);
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

  const entry = {
    id: metadata.id ?? timestamp,
    role,
    content,
    timestamp,
    origin: metadata.origin ?? 'floating',
    mode: metadata.mode ?? MODE_CHAT,
    ...metadata
  };

  assignTurnNumber(entry, metadata.turnNumber);
  const normalizedId = normalizeMessageId(entry.id);
  if (metadata.pinned || pinnedMessageIds.has(normalizedId)) {
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
  elements.memoryStatus.innerHTML = `Floating memory: ${used} / ${config.memoryLimitMB}&nbsp;MB • ${totalCount} ${messageLabel} (${pinnedCount} pinned)${ragLabel}`;
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

  const reflexLabel = config.reflexEnabled
    ? `enabled every ${config.reflexInterval} turn(s)`
    : 'disabled';
  diagnostics.push(`• Reflex mode: ${reflexLabel}`);
  const entropyDetails = describeEntropy(currentEntropyScore);
  diagnostics.push(
    `• Entropy (last ${config.entropyWindow} turns): ${(currentEntropyScore * 100).toFixed(0)}% ${entropyDetails.label}`
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

  const report = `Diagnostics @ ${startedAt.toLocaleTimeString()}\n${diagnostics.join('\n')}`;
  addSystemMessage(report);

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

async function retrieveRelevantMemories(query, limit) {
  const tokens = tokenize(query);
  if (!tokens.length) return [];

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

  const scores = [];
  candidates.forEach((value) => {
    const score = cosineSimilarity(tokens, tokenize(value.content));
    if (score > 0) {
      scores.push({ entry: value, score });
    }
  });

  scores.sort((a, b) => b.score - a.score);
  const selected = limit > 0 ? scores.slice(0, limit) : scores;
  return selected.map((item) => item.entry);
}

function tokenize(text) {
  return (text || '')
    .toLowerCase()
    .match(/[a-z0-9]+/g)
    ?.filter((token) => token.length > 1) || [];
}

function cosineSimilarity(queryTokens, docTokens) {
  if (!docTokens.length) return 0;
  const tf = new Map();
  for (const token of docTokens) {
    tf.set(token, (tf.get(token) || 0) + 1);
  }
  let dotProduct = 0;
  let queryMagnitude = 0;
  let docMagnitude = 0;
  const queryCounts = new Map();
  for (const token of queryTokens) {
    queryCounts.set(token, (queryCounts.get(token) || 0) + 1);
  }
  queryCounts.forEach((count, token) => {
    queryMagnitude += count * count;
    if (tf.has(token)) {
      dotProduct += count * tf.get(token);
    }
  });
  tf.forEach((count) => {
    docMagnitude += count * count;
  });
  if (!queryMagnitude || !docMagnitude) return 0;
  return dotProduct / Math.sqrt(queryMagnitude * docMagnitude);
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

  if (preset?.reasoningTimeoutMs && Number.isFinite(preset.reasoningTimeoutMs) && preset.reasoningTimeoutMs > 0) {
    reasoningMs = Math.max(reasoningMs, preset.reasoningTimeoutMs);
  }

  const normalizedModel = typeof modelId === 'string' ? modelId.toLowerCase() : '';
  const isReasoningModel = Boolean(preset?.anthropicFormat) ||
    (normalizedModel && REASONING_MODEL_HINTS.some((hint) => normalizedModel.includes(hint)));

  if (isReasoningModel) {
    return reasoningMs;
  }

  return baseMs;
}

async function callModel(messages, overrides = {}) {
  const endpoint = overrides.endpoint ?? config.endpoint;
  const model = overrides.model ?? config.model;
  const temperature = overrides.temperature ?? config.temperature;
  const apiKey = overrides.apiKey ?? config.apiKey;
  const maxTokens = overrides.maxTokens ?? config.maxResponseTokens;
  const providerPreset = overrides.providerPreset ?? config.providerPreset;

  if (!endpoint || !model) {
    throw new Error('Model endpoint or name missing.');
  }

  const preset = providerPresetMap.get(providerPreset);
  const timeoutMs = resolveRequestTimeout(model, preset, overrides.timeoutMs);
  const isAnthropic = preset?.anthropicFormat;
  const isGoogle = preset?.googleFormat;

  let headers = { 'Content-Type': 'application/json' };
  let payload = {};
  let requestEndpoint = endpoint;

  if (isAnthropic) {
    // Anthropic format
    headers = {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01'
    };
    payload = {
      model,
      max_tokens: Math.round(maxTokens) || 4096,
      temperature,
      messages: messages.map(msg => ({
        role: msg.role === 'assistant' ? 'assistant' : 'user',
        content: msg.content
      }))
    };
  } else if (isGoogle) {
    // Google Gemini format
    requestEndpoint = `${endpoint}${model}:generateContent?key=${apiKey}`;
    payload = {
      contents: [{
        parts: [{ text: messages.map(msg => `${msg.role}: ${msg.content}`).join('\n\n') }]
      }],
      generationConfig: {
        temperature,
        maxOutputTokens: Math.round(maxTokens) || 2048
      }
    };
  } else {
    // Standard OpenAI format
    if (apiKey) {
      headers.Authorization = `Bearer ${apiKey}`;
    }
    payload = {
      model,
      messages,
      temperature,
      stream: false
    };
    if (Number.isFinite(maxTokens) && maxTokens > 0) {
      payload.max_tokens = Math.round(maxTokens);
    }
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  let response;

  try {
    response = await fetch(requestEndpoint, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: controller.signal
    });
    clearTimeout(timeoutId);
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      const seconds = Math.round(timeoutMs / 100) / 10;
      throw new Error(`Request timed out after ${seconds}s.`);
    }
    throw error;
  }
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`HTTP ${response.status}: ${text}`);
  }

  const data = await response.json();

  let message = {};
  if (isAnthropic) {
    // Anthropic response format
    message = {
      content: data.content?.[0]?.text || '',
      reasoning: data.content?.find(c => c.type === 'thinking')?.thinking || ''
    };
  } else if (isGoogle) {
    // Google Gemini response format
    message = {
      content: data.candidates?.[0]?.content?.parts?.[0]?.text || ''
    };
  } else {
    // Standard OpenAI format
    message = data?.choices?.[0]?.message ?? {};
  }

  const { text, reasoning } = normalizeModelMessage(message);
  const segments = [];
  if (reasoning) {
    segments.push(`Reasoning:\n${reasoning}`);
  }
  if (text) {
    segments.push(text);
  }
  const combined = segments.join('\n\n').trim();
  if (!combined) {
    return { content: '', truncated: false, reasoning: reasoning ?? '' };
  }

  if (!Number.isFinite(maxTokens) || maxTokens <= 0) {
    return { content: combined, truncated: false, reasoning };
  }

  const trimmedResult = trimResponseToTokenLimit(combined, Math.round(maxTokens));
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
  const delaySeconds = clampNumber(
    config.dualTurnDelaySeconds ?? defaultConfig.dualTurnDelaySeconds,
    0,
    600,
    defaultConfig.dualTurnDelaySeconds
  );
  const delayMs = Math.max(0, Math.round(delaySeconds * 1000));
  if (delayMs === 0) {
    elements.dualStatus.textContent = 'Dual chat running…';
    autoContinueTimer = setTimeout(() => {
      autoContinueTimer = undefined;
      void advanceDualTurn();
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
    void advanceDualTurn();
  }, delayMs);
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

function computeEntropyScore(history, windowSize) {
  if (!Array.isArray(history) || history.length === 0) {
    return 0;
  }
  const size = clampNumber(windowSize ?? config.entropyWindow ?? defaultConfig.entropyWindow, 2, 50, 6);
  const recent = history.slice(-size);
  const tokens = recent
    .map((entry) => entry?.content ?? '')
    .join(' ')
    .toLowerCase()
    .match(/[\w\-']+/g);
  if (!tokens || tokens.length === 0) {
    return 0;
  }
  const unique = new Set(tokens);
  return unique.size / tokens.length;
}

function describeEntropy(score) {
  if (!Number.isFinite(score) || score <= 0) {
    return { state: 'calibrating', label: 'Calibrating…' };
  }
  if (score < 0.3) {
    return { state: 'loop', label: 'Looping' };
  }
  if (score < 0.55) {
    return { state: 'steady', label: 'Steady' };
  }
  return { state: 'explore', label: 'Exploring' };
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
    const lookback = Math.max(config.entropyWindow ?? 6, config.reflexInterval ?? 4, 6);
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
    const decorated = `Reflex summary\n${summary.trim()}`;
    recordDualMessage(agent, decorated, {
      marker: 'reflex',
      label: 'Reflex',
      countsAsTurn: false
    });
    lastReflexSummaryAt = Date.now();
    reflexSummaryCount += 1;
    void recordLog('arena', `${agentName} posted Reflex summary #${reflexSummaryCount}`, { silent: true });
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

async function generateDualTurn(speaker, seed, options = {}) {
  if (!isDualChatRunning) return;
  const partner = speaker === 'A' ? 'B' : 'A';
  const speakerName = getAgentDisplayName(speaker);
  const partnerName = getAgentDisplayName(partner);
  const normalizedSeed = typeof seed === 'string' && seed.trim() ? seed : activeDualSeed || config.dualSeed || DEFAULT_DUAL_SEED;
  const persona = getAgentPersona(speaker);
  const connection = activeDualConnections[speaker] ?? getAgentConnection(speaker);
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

  for (const turn of dualChatHistory) {
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
    retrievedMemories = await retrieveRelevantMemories(retrievalQuery, config.retrievalCount);
    if (retrievedMemories.length) {
      const compiled = retrievedMemories
        .map((item) => `${formatTimestamp(item.timestamp)} • ${item.role}: ${item.content}`)
        .join('\n');
      historyMessages.push({
        role: 'system',
        content: `Shared long-term memories:\n${compiled}`
      });
    }
  }
  if (autoInjecting) {
    registerRetrieval(speaker, retrievedMemories.length);
  }

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
    const { content: reply, truncated, reasoning } = await callModel(historyMessages, connection);
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
      label: badgeParts.length ? badgeParts.join(' • ') : undefined
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
  const { saveHistory = true, label, marker, turnNumber: providedTurn, countsAsTurn = true } = options;
  const timestamp = Date.now();
  let turnNumber = typeof providedTurn === 'number' ? providedTurn : null;

  const speakerName = getAgentDisplayName(speaker);

  if (saveHistory) {
    if (countsAsTurn) {
      if (speaker === 'A') {
        dualTurnCounter += 1;
        if (turnNumber === null) {
          turnNumber = dualTurnCounter;
        }
      } else {
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
    dualChatHistory.push({ speaker, content, timestamp, turnNumber });

    const memoryEntry = {
      id: `arena-${timestamp}-${speaker}-${dualChatHistory.length}`,
      role: speakerName,
      speaker,
      speakerName,
      content,
      timestamp,
      origin: 'arena',
      turnNumber,
      mode: MODE_ARENA
    };
    floatingMemory.push(memoryEntry);
    trimFloatingMemory();
    queueMemoryCheckpoint(MODE_ARENA, memoryEntry);
    void persistMessage(memoryEntry);
  }

  const node = elements.dualMessageTemplate.content.firstElementChild.cloneNode(true);
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
  node.querySelector('.dual-time').textContent = formatTimestamp(timestamp);
  node.querySelector('.dual-content').textContent = content;
  if (marker) {
    node.classList.add(`dual-message--${marker}`);
  }
  elements.dualChatWindow.appendChild(node);
  scrollContainerToBottom(elements.dualChatWindow);
  void persistDualRagSnapshot();
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
