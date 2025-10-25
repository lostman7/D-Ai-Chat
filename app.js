import { boardEntriesToChunks, applyBoardEmbeddings, vectorSearch } from './chunker.js';
import {
  initVectorCache,
  getCachedVectors,
  storeCachedVectors,
  subscribeVectorCache,
  VECTOR_CACHE_LIMIT_BYTES
} from './rag/vectorCache.js';

const STORAGE_KEY = 'sam_settings_v2';
const BOARD_STORAGE_KEY = 'sam_board_store_v1';
const BOARD_FILE_PATH = 'arena/board.json';
const ARENA_CONFIG_PATH = 'arena/config.json';
const AGENT_CONFIG_DIR = 'arena/agents';
const FLOATING_MEMORY_KEY = 'sam_floating_memory_v1';
const AGENT_CACHE_PREFIX = 'sam_agent_cache_';
const OLLAMA_TAGS_URL = 'http://localhost:11434/api/tags';
const OLLAMA_EMBED_REGEX = /embed|embedding/i;
const MB = 1024 * 1024;
const textEncoder = typeof TextEncoder !== 'undefined' ? new TextEncoder() : null;

const DEFAULT_AGENT_TEMPLATE = {
  model: {
    provider: 'ollama_local',
    preset: 'default',
    endpoint: 'http://localhost:11434/api/generate',
    name: '',
    key: '',
    temp: 0.7,
    maxTokens: 2048
  },
  embed: {
    provider: 'ollama_local',
    endpoint: 'http://localhost:11434/api/embed',
    model: '',
    key: '',
    ctx: 2048
  },
  persona: ''
};

function deepClone(value) {
  if (typeof structuredClone === 'function') {
    try {
      return structuredClone(value);
    } catch (error) {
      // fall through
    }
  }
  return JSON.parse(JSON.stringify(value));
}

const DEFAULT_SETTINGS = {
  arena: {
    mode: 'human',
    turnCooldownMs: 15000,
    perAgentCooldown: {},
    parallel: false,
    boardPolicy: 'last_k_messages',
    boardWindow: 20,
    summaryEvery: 5
  },
  memory: {
    budgetMb: 100,
    retrievalCount: 0,
    autoInject: false,
    useEmbeddings: false
  },
  agents: {
    A: createAgent('A'),
    B: createAgent('B')
  }
};

const DEFAULT_BOARD = {
  entries: [],
  lastTurn: 0,
  lastRound: 0,
  summaryTurn: 0
};

const providerDefaults = {
  ollama_local: {
    chatEndpoint: 'http://localhost:11434/api/generate',
    embedEndpoint: 'http://localhost:11434/api/embed'
  },
  ollama_cloud: {
    chatEndpoint: 'https://api.ollama.ai/v1/chat/completions',
    embedEndpoint: 'https://api.ollama.ai/v1/embeddings'
  },
  lmstudio: {
    chatEndpoint: 'http://127.0.0.1:1234/v1/chat/completions',
    embedEndpoint: 'http://127.0.0.1:1234/v1/embeddings'
  }
};

const chatProviders = new Set(['ollama_local', 'ollama_cloud', 'lmstudio']);
const humanAgentId = 'human';
const boardPersistDebounceMs = 1000;
const chatTimeoutMs = 60000;
const embedTimeoutMs = 30000;

let floatingMemory = [];
const agentCaches = new Map();
const agentCachePersisting = new Map();
const ragStats = { totalEntries: 0, totalBytes: 0 };
let memoryEventsBound = false;

let settings = deepClone(DEFAULT_SETTINGS);
let boardState = deepClone(DEFAULT_BOARD);
let arenaState = createArenaState();
let modelCatalog = { timestamp: 0, models: [], embeds: [] };
let ollamaRefreshTimer = null;
let accordionBound = false;
let pendingBoardPersist = null;
let humanInputResolver = null;
const humanInputQueue = [];

const elements = {};
let unsubscribeVectorCache = null;

function createAgent(id) {
  const agent = deepClone(DEFAULT_AGENT_TEMPLATE);
  agent.id = id;
  return agent;
}

function resolveAgentId(agent) {
  if (agent?.id) return agent.id;
  for (const [id, value] of Object.entries(settings.agents)) {
    if (value === agent) return id;
  }
  return 'unknown';
}

function createArenaState() {
  return {
    running: false,
    paused: false,
    mode: 'human',
    workers: new Map(),
    abortController: null,
    turn: 0,
    round: 1,
    lastByAgent: new Map(),
    order: [],
    boardWriteLock: false
  };
}

function getStorage() {
  if (typeof window !== 'undefined' && window.standaloneStore) {
    const bridge = window.standaloneStore;
    return {
      getItem: (key) => bridge.getItem(key),
      setItem: (key, value) => bridge.setItem(key, value),
      removeItem: (key) => bridge.removeItem(key)
    };
  }
  if (typeof window !== 'undefined' && window.localStorage) {
    return window.localStorage;
  }
  const memory = new Map();
  return {
    getItem: (key) => (memory.has(key) ? memory.get(key) : null),
    setItem: (key, value) => memory.set(key, value),
    removeItem: (key) => memory.delete(key)
  };
}

const storage = getStorage();

function loadSettings() {
  try {
    const raw = storage.getItem(STORAGE_KEY);
    if (!raw) {
      settings = deepClone(DEFAULT_SETTINGS);
      return;
    }
    const parsed = JSON.parse(raw);
    settings = migrateSettings(parsed);
  } catch (error) {
    console.error('Failed to load settings, using defaults', error);
    settings = deepClone(DEFAULT_SETTINGS);
  }
}

function migrateSettings(value) {
  const base = deepClone(DEFAULT_SETTINGS);
  if (!value || typeof value !== 'object') {
    return base;
  }
  const arena = value.arena && typeof value.arena === 'object' ? value.arena : {};
  base.arena = {
    mode: arena.mode ?? 'human',
    turnCooldownMs: Number.isFinite(Number(arena.turnCooldownMs)) ? Number(arena.turnCooldownMs) : 15000,
    perAgentCooldown: arena.perAgentCooldown && typeof arena.perAgentCooldown === 'object' ? { ...arena.perAgentCooldown } : {},
    parallel: Boolean(arena.parallel),
    boardPolicy: arena.boardPolicy ?? 'last_k_messages',
    boardWindow: Number.isFinite(Number(arena.boardWindow)) ? Number(arena.boardWindow) : 20,
    summaryEvery: Number.isFinite(Number(arena.summaryEvery)) ? Number(arena.summaryEvery) : 5
  };

  const memory = value.memory && typeof value.memory === 'object' ? value.memory : {};
  const legacyBudget = Number.isFinite(Number(value.memoryBudgetMb)) ? Number(value.memoryBudgetMb) : null;
  const legacyRetrieval = Number.isFinite(Number(value.memoriesPerRetrieval)) ? Number(value.memoriesPerRetrieval) : null;
  base.memory = {
    budgetMb: Number.isFinite(Number(memory.budgetMb))
      ? Number(memory.budgetMb)
      : legacyBudget || 100,
    retrievalCount: Number.isFinite(Number(memory.retrievalCount))
      ? Number(memory.retrievalCount)
      : legacyRetrieval || 0,
    autoInject: typeof memory.autoInject === 'boolean' ? memory.autoInject : Boolean(value.autoInjectMemories),
    useEmbeddings:
      typeof memory.useEmbeddings === 'boolean'
        ? memory.useEmbeddings
        : Boolean(value.embeddingRetrieval ?? value.useEmbeddingRetrieval)
  };

  const agents = {};
  if (value.agents && typeof value.agents === 'object') {
    for (const [key, agentValue] of Object.entries(value.agents)) {
      if (!/^[A-G]$/.test(key)) continue;
      agents[key] = mergeAgent(agentValue, key);
    }
  }

  if (!Object.keys(agents).length) {
    const legacyA = value.agentA ?? value.A ?? value.agentSettingsA;
    const legacyB = value.agentB ?? value.B ?? value.agentSettingsB;
    agents.A = mergeAgent(legacyA, 'A');
    agents.B = mergeAgent(legacyB, 'B');
  }

  base.agents = Object.keys(agents).length ? agents : { A: createAgent('A'), B: createAgent('B') };
  return base;
}

function mergeAgent(value, id) {
  const template = createAgent(id);
  if (!value || typeof value !== 'object') {
    return template;
  }
  const result = deepClone(template);
  if (value.model && typeof value.model === 'object') {
    result.model.provider = value.model.provider ?? result.model.provider;
    result.model.preset = value.model.preset ?? result.model.preset;
    result.model.endpoint = value.model.endpoint ?? providerDefaults[result.model.provider]?.chatEndpoint ?? result.model.endpoint;
    result.model.name = value.model.name ?? result.model.name;
    result.model.key = value.model.key ?? result.model.key;
    result.model.temp = Number.isFinite(Number(value.model.temp)) ? Number(value.model.temp) : result.model.temp;
    result.model.maxTokens = Number.isFinite(Number(value.model.maxTokens))
      ? Number(value.model.maxTokens)
      : result.model.maxTokens;
  }
  if (value.embed && typeof value.embed === 'object') {
    result.embed.provider = value.embed.provider ?? result.embed.provider;
    result.embed.endpoint = value.embed.endpoint ?? providerDefaults[result.embed.provider]?.embedEndpoint ?? result.embed.endpoint;
    result.embed.model = value.embed.model ?? result.embed.model;
    result.embed.key = value.embed.key ?? result.embed.key;
    result.embed.ctx = Number.isFinite(Number(value.embed.ctx)) ? Number(value.embed.ctx) : result.embed.ctx;
  }
  if (typeof value.persona === 'string') {
    result.persona = value.persona;
  }
  result.id = id;
  return result;
}

function persistSettings() {
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify(settings));
    void persistArenaConfig();
  } catch (error) {
    console.error('Failed to persist settings', error);
  }
}

function createBoardStore() {
  if (typeof window === 'undefined') {
    let memoryBoard = deepClone(DEFAULT_BOARD);
    return {
      async load() {
        return deepClone(memoryBoard);
      },
      async save(data) {
        memoryBoard = deepClone(data);
      }
    };
  }
  const fs = window.standaloneFs;
  const store = window.standaloneStore;
  return {
    async load() {
      if (fs && typeof fs.readJson === 'function') {
        try {
          const payload = await fs.readJson(BOARD_FILE_PATH);
          return normalizeBoard(payload);
        } catch (error) {
          console.warn('Failed to read arena board from filesystem, falling back to cache', error);
        }
      }
      const raw = store?.getItem?.(BOARD_STORAGE_KEY) ?? storage.getItem(BOARD_STORAGE_KEY);
      if (!raw) return deepClone(DEFAULT_BOARD);
      try {
        return normalizeBoard(JSON.parse(raw));
      } catch (error) {
        console.error('Invalid board cache payload', error);
        return deepClone(DEFAULT_BOARD);
      }
    },
    async save(data) {
      const payload = normalizeBoard(data);
      if (fs && typeof fs.writeJson === 'function') {
        try {
          await fs.writeJson(BOARD_FILE_PATH, payload);
        } catch (error) {
          console.warn('Failed to write arena board to filesystem, falling back to cache', error);
        }
      }
      try {
        storage.setItem(BOARD_STORAGE_KEY, JSON.stringify(payload));
      } catch (error) {
        console.error('Failed to persist board cache', error);
      }
    }
  };
}

const boardStore = createBoardStore();
const floatingMemoryStore = createFloatingMemoryStore();

function createFloatingMemoryStore() {
  if (typeof window === 'undefined') {
    let memory = [];
    return {
      async load() {
        return memory.map(normalizeMemoryEntry);
      },
      async save(entries) {
        memory = entries.map(normalizeMemoryEntry);
      }
    };
  }
  return {
    async load() {
      const raw = storage.getItem(FLOATING_MEMORY_KEY);
      if (!raw) return [];
      try {
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed.map(normalizeMemoryEntry) : [];
      } catch (error) {
        console.warn('Failed to parse floating memory store', error);
        return [];
      }
    },
    async save(entries) {
      try {
        storage.setItem(FLOATING_MEMORY_KEY, JSON.stringify(entries));
      } catch (error) {
        console.error('Failed to persist floating memory', error);
      }
    }
  };
}

function normalizeMemoryEntry(entry) {
  if (!entry || typeof entry !== 'object') {
    return {
      id: `memory-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      content: '',
      ts: new Date().toISOString(),
      pinned: false,
      agentId: null,
      tokens: 0,
      embeddings: {}
    };
  }
  return {
    id: entry.id ?? `memory-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    content: entry.content ?? '',
    ts: entry.ts ?? new Date().toISOString(),
    pinned: Boolean(entry.pinned),
    agentId: entry.agentId ?? null,
    turn: entry.turn ?? null,
    round: entry.round ?? null,
    tokens: Number.isFinite(Number(entry.tokens)) ? Number(entry.tokens) : estimateTokens(entry.content ?? ''),
    embeddings: entry.embeddings && typeof entry.embeddings === 'object' ? { ...entry.embeddings } : {}
  };
}

async function loadFloatingMemory() {
  try {
    floatingMemory = (await floatingMemoryStore.load()).map(normalizeMemoryEntry);
  } catch (error) {
    console.warn('Failed to load floating memory, starting fresh', error);
    floatingMemory = [];
  }
  maintainMemoryBudget();
  updateMemoryStats();
}

async function persistFloatingMemory() {
  try {
    await floatingMemoryStore.save(floatingMemory);
  } catch (error) {
    console.error('Failed to persist floating memory', error);
  }
}

function entryBytes(entry) {
  if (!entry || !entry.content) return 0;
  if (textEncoder) {
    try {
      return textEncoder.encode(entry.content).length;
    } catch (error) {
      return entry.content.length * 2;
    }
  }
  return entry.content.length * 2;
}

function getMemoryUsageBytes() {
  return floatingMemory.reduce((total, entry) => total + entryBytes(entry), 0);
}

function maintainMemoryBudget() {
  const limit = Math.max(50, Number(settings.memory?.budgetMb ?? 100)) * MB;
  let usage = getMemoryUsageBytes();
  if (usage <= limit) {
    return;
  }
  for (let index = 0; index < floatingMemory.length && usage > limit; index += 1) {
    const entry = floatingMemory[index];
    if (entry.pinned) continue;
    usage -= entryBytes(entry);
    floatingMemory.splice(index, 1);
    index -= 1;
  }
}

function trackMemoryEntry(entry) {
  if (!entry || !entry.content || entry.summary) {
    return;
  }
  const normalized = normalizeMemoryEntry(entry);
  const existingIndex = floatingMemory.findIndex((item) => item.id === normalized.id);
  if (existingIndex >= 0) {
    floatingMemory[existingIndex] = { ...floatingMemory[existingIndex], ...normalized };
  } else {
    floatingMemory.push(normalized);
  }
  maintainMemoryBudget();
  void persistFloatingMemory();
  renderFloatingMemory();
}

function removeMemoryEntry(id) {
  const index = floatingMemory.findIndex((entry) => entry.id === id);
  if (index === -1) return;
  floatingMemory.splice(index, 1);
  void persistFloatingMemory();
  renderFloatingMemory();
}

function toggleMemoryPinned(id) {
  const target = floatingMemory.find((entry) => entry.id === id);
  if (!target) return;
  target.pinned = !target.pinned;
  void persistFloatingMemory();
  renderFloatingMemory();
}

function archiveUnpinnedMemory() {
  const before = floatingMemory.length;
  floatingMemory = floatingMemory.filter((entry) => entry.pinned);
  if (floatingMemory.length !== before) {
    void persistFloatingMemory();
    renderFloatingMemory();
    showToast(`Archived ${before - floatingMemory.length} memory items.`);
  } else {
    showToast('No unpinned memories to archive.');
  }
}

function renderFloatingMemory() {
  if (!elements.floatingMemoryList) return;
  elements.floatingMemoryList.innerHTML = '';
  const fragment = document.createDocumentFragment();
  for (const entry of floatingMemory) {
    fragment.appendChild(renderMemoryItem(entry));
  }
  elements.floatingMemoryList.appendChild(fragment);
  updateMemoryStats();
  updateMemoryStatus();
}

function renderMemoryItem(entry) {
  const item = document.createElement('article');
  item.className = 'memory-item';
  item.dataset.id = entry.id;
  item.dataset.pinned = String(Boolean(entry.pinned));
  const meta = document.createElement('div');
  meta.className = 'memory-item__meta';
  const label = entry.agentId ? formatAgentLabel(entry.agentId) : 'Memory';
  meta.textContent = `${label} • ${new Date(entry.ts).toLocaleString()}`;
  const content = document.createElement('p');
  content.className = 'memory-item__content';
  content.textContent = entry.content;
  const actions = document.createElement('div');
  actions.className = 'memory-item__actions';
  const pin = document.createElement('button');
  pin.type = 'button';
  pin.className = 'ghost';
  pin.dataset.action = 'toggle-pin';
  pin.textContent = entry.pinned ? 'Unpin' : 'Pin';
  const remove = document.createElement('button');
  remove.type = 'button';
  remove.className = 'ghost danger';
  remove.dataset.action = 'remove';
  remove.textContent = 'Remove';
  actions.append(pin, remove);
  item.append(meta, content, actions);
  return item;
}

function updateMemoryStats() {
  const total = floatingMemory.length;
  const pinned = floatingMemory.filter((entry) => entry.pinned).length;
  const usageMb = getMemoryUsageBytes() / MB;
  if (elements.floatingMemoryCount) {
    elements.floatingMemoryCount.textContent = String(total);
  }
  if (elements.pinnedMemoryCount) {
    elements.pinnedMemoryCount.textContent = String(pinned);
  }
  if (elements.memoryStatus) {
    const budget = Math.max(50, Number(settings.memory?.budgetMb ?? 100));
    elements.memoryStatus.textContent = `Floating memory: ${usageMb.toFixed(1)} / ${budget} MB`;
  }
  updateRagStats();
}

function updateMemoryStatus() {
  if (!elements.memorySliderValue) return;
  const budget = Math.max(50, Number(settings.memory?.budgetMb ?? 100));
  elements.memorySliderValue.textContent = `${budget} MB`;
}

function exportFloatingMemory() {
  const payload = {
    generatedAt: new Date().toISOString(),
    budgetMb: settings.memory?.budgetMb ?? 100,
    entries: floatingMemory
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `sam-floating-memory-${Date.now()}.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function getAgentCacheKey(agentId) {
  return `${AGENT_CACHE_PREFIX}${agentId}`;
}

function getAgentCachePath(agentId) {
  return `rag/${agentId.toLowerCase()}_cache/cache.json`;
}

function normalizeAgentCache(value) {
  if (!value || typeof value !== 'object') {
    return { entries: [] };
  }
  const entries = Array.isArray(value.entries) ? value.entries.filter(Boolean) : [];
  return {
    entries: entries.map((entry) => ({
      id: entry.id ?? `cache-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      content: entry.content ?? '',
      ts: entry.ts ?? new Date().toISOString(),
      tokens: Number.isFinite(Number(entry.tokens)) ? Number(entry.tokens) : estimateTokens(entry.content ?? ''),
      embedding: Array.isArray(entry.embedding) ? entry.embedding : null,
      round: entry.round ?? null,
      turn: entry.turn ?? null
    }))
  };
}

async function loadAgentCache(agentId) {
  if (agentCaches.has(agentId)) {
    return agentCaches.get(agentId);
  }
  let cache = { entries: [] };
  if (typeof window !== 'undefined') {
    const fs = window.standaloneFs;
    if (fs && typeof fs.readJson === 'function') {
      try {
        const payload = await fs.readJson(getAgentCachePath(agentId));
        cache = normalizeAgentCache(payload);
      } catch (error) {
        if (error?.code !== 'ENOENT') {
          console.warn(`Failed to read cache for agent ${agentId} from disk`, error);
        }
      }
    }
  }
  if (!cache.entries.length) {
    const raw = storage.getItem(getAgentCacheKey(agentId));
    if (raw) {
      try {
        cache = normalizeAgentCache(JSON.parse(raw));
      } catch (error) {
        console.warn('Failed to parse cached agent memory from storage', error);
      }
    }
  }
  agentCaches.set(agentId, cache);
  updateRagStats();
  return cache;
}

async function persistAgentCache(agentId) {
  const cache = agentCaches.get(agentId) ?? { entries: [] };
  const payload = normalizeAgentCache(cache);
  if (agentCachePersisting.has(agentId)) {
    return agentCachePersisting.get(agentId);
  }
  const task = (async () => {
    if (typeof window !== 'undefined') {
      const fs = window.standaloneFs;
      if (fs && typeof fs.writeJson === 'function') {
        try {
          const path = getAgentCachePath(agentId);
          const parts = path.split('/');
          if (parts.length > 1 && typeof fs.ensureDir === 'function') {
            await fs.ensureDir(parts.slice(0, -1).join('/'));
          }
          await fs.writeJson(path, payload);
        } catch (error) {
          console.warn(`Failed to persist agent cache for ${agentId} to disk`, error);
        }
      }
    }
    try {
      storage.setItem(getAgentCacheKey(agentId), JSON.stringify(payload));
    } catch (error) {
      console.error(`Failed to persist agent cache for ${agentId}`, error);
    }
    agentCachePersisting.delete(agentId);
  })();
  agentCachePersisting.set(agentId, task);
  return task;
}

async function appendAgentCache(agentId, entry) {
  if (!settings.agents[agentId] || !entry?.content) return;
  const cache = await loadAgentCache(agentId);
  const record = {
    id: entry.id ?? `cache-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    content: entry.content,
    ts: entry.ts ?? new Date().toISOString(),
    tokens: entry.tokens ?? estimateTokens(entry.content),
    round: entry.round ?? null,
    turn: entry.turn ?? null,
    embedding: null
  };
  if (settings.memory.useEmbeddings) {
    try {
      const vectors = await maybeEmbed(settings.agents[agentId], [entry.content]);
      record.embedding = vectors?.[0] ?? null;
    } catch (error) {
      console.warn('Agent cache embedding failed', error);
    }
  }
  cache.entries.push(record);
  const maxEntries = 200;
  if (cache.entries.length > maxEntries) {
    cache.entries.splice(0, cache.entries.length - maxEntries);
  }
  agentCaches.set(agentId, cache);
  updateRagStats();
  await persistAgentCache(agentId);
}

function updateRagStats() {
  let totalEntries = 0;
  let totalBytes = 0;
  for (const cache of agentCaches.values()) {
    if (!cache || !Array.isArray(cache.entries)) continue;
    totalEntries += cache.entries.length;
    for (const entry of cache.entries) {
      totalBytes += entryBytes(entry);
    }
  }
  ragStats.totalEntries = totalEntries;
  ragStats.totalBytes = totalBytes;
  if (elements.ragMemoryCount) {
    elements.ragMemoryCount.textContent = String(totalEntries);
  }
  if (elements.ragFootprint) {
    elements.ragFootprint.textContent = `${(totalBytes / MB).toFixed(1)} MB`;
  }
  if (elements.ragImportStatus) {
    elements.ragImportStatus.textContent = totalEntries
      ? `Loaded ${totalEntries} cached embeddings.`
      : 'RAG archives not loaded yet.';
  }
}

function normalizeBoard(value) {
  if (!value || typeof value !== 'object') {
    return deepClone(DEFAULT_BOARD);
  }
  const entries = Array.isArray(value.entries) ? value.entries.filter(Boolean) : [];
  return {
    entries,
    lastTurn: Number.isFinite(Number(value.lastTurn)) ? Number(value.lastTurn) : entries.length,
    lastRound: Number.isFinite(Number(value.lastRound)) ? Number(value.lastRound) : 0,
    summaryTurn: Number.isFinite(Number(value.summaryTurn)) ? Number(value.summaryTurn) : 0
  };
}

function bindOptionsHandle() {
  const handle = document.getElementById('optionsHandle');
  const backdrop = document.getElementById('optionsBackdrop');
  const drawer = document.getElementById('optionsDrawer');
  function toggle(open) {
    const next = typeof open === 'boolean' ? open : !document.body.classList.contains('options-open');
    document.body.classList.toggle('options-open', next);
    if (drawer) {
      drawer.setAttribute('aria-hidden', String(!next));
      if (next) {
        drawer.removeAttribute('inert');
        drawer.focus?.();
      } else {
        drawer.setAttribute('inert', '');
      }
    }
    if (handle) {
      handle.setAttribute('aria-expanded', String(next));
    }
    if (backdrop) {
      backdrop.hidden = !next;
    }
  }
  handle?.addEventListener('click', () => toggle());
  backdrop?.addEventListener('click', () => toggle(false));
}

function populateReferences() {
  elements.arenaMode = document.getElementById('arena_mode');
  elements.turnCooldown = document.getElementById('turn_cooldown_ms');
  elements.boardPolicy = document.getElementById('board_policy');
  elements.boardWindow = document.getElementById('board_window');
  elements.summaryEvery = document.getElementById('summary_every');
  elements.summaryEveryField = document.getElementById('summaryEveryField');
  elements.parallelToggle = document.getElementById('parallel_mode_toggle');
  elements.memorySlider = document.getElementById('memorySlider');
  elements.memorySliderValue = document.getElementById('memorySliderValue');
  elements.memoryStatus = document.getElementById('memoryStatus');
  elements.retrievalCount = document.getElementById('retrievalCount');
  elements.autoInjectMemories = document.getElementById('autoInjectMemories');
  elements.embeddingToggle = document.getElementById('embeddingRetrievalToggle');
  elements.embeddingStatus = document.getElementById('embeddingStatus');
  elements.vectorCacheStatus = document.getElementById('vectorCacheStatus');
  elements.floatingMemoryList = document.getElementById('floatingMemoryList');
  elements.floatingMemoryCount = document.getElementById('floatingMemoryCount');
  elements.pinnedMemoryCount = document.getElementById('pinnedMemoryCount');
  elements.ragMemoryCount = document.getElementById('ragMemoryCount');
  elements.ragFootprint = document.getElementById('ragFootprint');
  elements.ragImportStatus = document.getElementById('ragImportStatus');
  elements.refreshFloatingButton = document.getElementById('refreshFloatingButton');
  elements.archiveUnpinnedButton = document.getElementById('archiveUnpinnedButton');
  elements.exportMemoryButton = document.getElementById('exportMemoryButton');
  elements.agentList = document.getElementById('agent_list');
  elements.addAgentBtn = document.getElementById('add_agent_btn');
  elements.boardViewport = document.getElementById('boardViewport');
  elements.boardStatus = document.getElementById('boardTurnStatus');
  elements.startArena = document.getElementById('startArenaBtn');
  elements.stopArena = document.getElementById('stopArenaBtn');
  elements.pauseArena = document.getElementById('pauseArenaBtn');
  elements.resumeArena = document.getElementById('resumeArenaBtn');
  elements.snapshotBoard = document.getElementById('snapshotBoardBtn');
  elements.exportBoard = document.getElementById('exportBoardBtn');
  elements.humanToggle = document.getElementById('humanInputToggle');
  elements.chatPanel = document.getElementById('chatPanel');
  elements.chatWindow = document.getElementById('chatWindow');
  elements.messageInput = document.getElementById('messageInput');
  elements.sendButton = document.getElementById('sendButton');
  elements.clearChatButton = document.getElementById('clearChatButton');
}

function renderArenaSettings() {
  if (elements.arenaMode) {
    elements.arenaMode.value = settings.arena.mode;
  }
  if (elements.turnCooldown) {
    elements.turnCooldown.value = settings.arena.turnCooldownMs;
  }
  if (elements.boardPolicy) {
    elements.boardPolicy.value = settings.arena.boardPolicy;
  }
  if (elements.boardWindow) {
    elements.boardWindow.value = settings.arena.boardWindow;
  }
  if (elements.summaryEvery) {
    elements.summaryEvery.value = settings.arena.summaryEvery;
  }
  if (elements.parallelToggle) {
    elements.parallelToggle.checked = settings.arena.parallel;
  }
  updateSummaryVisibility();
}

function renderMemorySettings() {
  if (elements.memorySlider) {
    elements.memorySlider.value = settings.memory.budgetMb;
  }
  if (elements.retrievalCount) {
    elements.retrievalCount.value = settings.memory.retrievalCount;
  }
  if (elements.autoInjectMemories) {
    elements.autoInjectMemories.checked = Boolean(settings.memory.autoInject);
  }
  if (elements.embeddingToggle) {
    elements.embeddingToggle.checked = Boolean(settings.memory.useEmbeddings);
  }
  updateEmbeddingStatus();
  updateVectorCacheStatus();
  updateMemoryStatus();
  updateMemoryStats();
}

function updateSummaryVisibility() {
  if (!elements.summaryEveryField) return;
  const show = settings.arena.boardPolicy === 'summary_every_n_turns';
  elements.summaryEveryField.style.display = show ? '' : 'none';
}

function bindArenaSettings() {
  elements.arenaMode?.addEventListener('change', (event) => {
    settings.arena.mode = event.target.value;
    renderArenaSettings();
    persistSettings();
  });
  elements.turnCooldown?.addEventListener('change', (event) => {
    const value = Number(event.target.value);
    settings.arena.turnCooldownMs = Number.isFinite(value) && value >= 0 ? value : 0;
    persistSettings();
  });
  elements.boardPolicy?.addEventListener('change', (event) => {
    settings.arena.boardPolicy = event.target.value;
    updateSummaryVisibility();
    persistSettings();
  });
  elements.boardWindow?.addEventListener('change', (event) => {
    const value = Number(event.target.value);
    settings.arena.boardWindow = Number.isFinite(value) && value > 0 ? value : 1;
    persistSettings();
  });
  elements.summaryEvery?.addEventListener('change', (event) => {
    const value = Number(event.target.value);
    settings.arena.summaryEvery = Number.isFinite(value) && value > 0 ? value : 1;
    persistSettings();
  });
  elements.parallelToggle?.addEventListener('change', (event) => {
    settings.arena.parallel = Boolean(event.target.checked);
    persistSettings();
  });
}

function bindMemorySettings() {
  elements.memorySlider?.addEventListener('input', (event) => {
    const value = Number(event.target.value);
    settings.memory.budgetMb = Number.isFinite(value) ? value : settings.memory.budgetMb;
    updateMemoryStatus();
    maintainMemoryBudget();
    renderFloatingMemory();
    persistSettings();
    void persistFloatingMemory();
  });
  elements.retrievalCount?.addEventListener('change', (event) => {
    const value = Number(event.target.value);
    settings.memory.retrievalCount = Number.isFinite(value) && value >= 0 ? value : 0;
    persistSettings();
  });
  elements.autoInjectMemories?.addEventListener('change', (event) => {
    settings.memory.autoInject = Boolean(event.target.checked);
    persistSettings();
  });
  elements.embeddingToggle?.addEventListener('change', (event) => {
    settings.memory.useEmbeddings = Boolean(event.target.checked);
    updateEmbeddingStatus();
    persistSettings();
  });
  elements.exportMemoryButton?.addEventListener('click', () => {
    exportFloatingMemory();
  });
  elements.refreshFloatingButton?.addEventListener('click', () => {
    renderFloatingMemory();
  });
  elements.archiveUnpinnedButton?.addEventListener('click', () => {
    archiveUnpinnedMemory();
  });
  if (!memoryEventsBound && elements.floatingMemoryList) {
    memoryEventsBound = true;
    elements.floatingMemoryList.addEventListener('click', (event) => {
      const button = event.target.closest('button[data-action]');
      if (!button) return;
      const item = button.closest('.memory-item');
      if (!item) return;
      const id = item.dataset.id;
      if (!id) return;
      const action = button.dataset.action;
      if (action === 'toggle-pin') {
        toggleMemoryPinned(id);
      } else if (action === 'remove') {
        removeMemoryEntry(id);
      }
    });
  }
}

function updateEmbeddingStatus() {
  if (!elements.embeddingStatus) return;
  const state = settings.memory.useEmbeddings;
  elements.embeddingStatus.textContent = `Embedding Active ${state ? '✓ (enabled)' : '✗ (disabled)'}`;
}

function updateVectorCacheStatus(stats) {
  if (!elements.vectorCacheStatus) return;
  const fallbackSupported = typeof indexedDB !== 'undefined';
  const snapshot = stats ?? {
    bytes: 0,
    entries: 0,
    status: fallbackSupported ? 'Initialising…' : 'Unavailable',
    supported: fallbackSupported
  };
  if (!snapshot.supported) {
    elements.vectorCacheStatus.textContent = 'Vector cache unavailable in this environment.';
    return;
  }
  const sizeMb = (snapshot.bytes / MB).toFixed(1);
  const limitMb = (VECTOR_CACHE_LIMIT_BYTES / MB).toFixed(0);
  const entries = snapshot.entries ?? 0;
  const status = snapshot.status ?? 'Idle';
  elements.vectorCacheStatus.textContent = `Vector cache: ${sizeMb} MB of ${limitMb} MB • ${entries} entries • Sync: ${status}`;
}

function renderAgents() {
  if (!elements.agentList) return;
  elements.agentList.innerHTML = '';
  const template = document.getElementById('agentCardTemplate');
  const sorted = Object.keys(settings.agents).sort();
  for (const id of sorted) {
    const data = settings.agents[id];
    const fragment = template.content.cloneNode(true);
    const card = fragment.querySelector('.agent-card');
    card.dataset.agentId = id;
    const title = fragment.querySelector('.agent-card__title');
    title.textContent = `Agent ${id}`;
    const modelProvider = fragment.querySelector('.agent-model-provider');
    const modelPreset = fragment.querySelector('.agent-model-preset');
    const modelEndpoint = fragment.querySelector('.agent-model-endpoint');
    const modelName = fragment.querySelector('.agent-model-name');
    const modelKey = fragment.querySelector('.agent-model-key');
    const modelTemp = fragment.querySelector('.agent-model-temp');
    const modelMax = fragment.querySelector('.agent-model-max');
    const embedProvider = fragment.querySelector('.agent-embed-provider');
    const embedEndpoint = fragment.querySelector('.agent-embed-endpoint');
    const embedModel = fragment.querySelector('.agent-embed-model');
    const embedKey = fragment.querySelector('.agent-embed-key');
    const embedCtx = fragment.querySelector('.agent-embed-ctx');
    const persona = fragment.querySelector('.agent-persona');
    const removeBtn = fragment.querySelector('.agent-remove');
    const testChatBtn = fragment.querySelector('.agent-test-chat');
    const testEmbedBtn = fragment.querySelector('.agent-test-embed');
    const cooldownToggle = fragment.querySelector('.agent-cooldown-check');
    const cooldownField = fragment.querySelector('[data-per-agent-cooldown]');
    const cooldownInput = fragment.querySelector('.agent-cooldown-input');

    if (modelProvider) modelProvider.value = data.model.provider;
    if (modelPreset) modelPreset.value = data.model.preset ?? '';
    if (modelEndpoint) modelEndpoint.value = data.model.endpoint ?? '';
    if (modelName) modelName.value = data.model.name ?? '';
    if (modelKey) modelKey.value = data.model.key ?? '';
    if (modelTemp) modelTemp.value = data.model.temp ?? 0.7;
    if (modelMax) modelMax.value = data.model.maxTokens ?? 2048;
    if (embedProvider) embedProvider.value = data.embed.provider;
    if (embedEndpoint) embedEndpoint.value = data.embed.endpoint ?? '';
    if (embedModel) embedModel.value = data.embed.model ?? '';
    if (embedKey) embedKey.value = data.embed.key ?? '';
    if (embedCtx) embedCtx.value = data.embed.ctx ?? 2048;
    if (persona) persona.value = data.persona ?? '';
    if (cooldownToggle) {
      const perAgent = settings.arena.perAgentCooldown?.[id] ?? 0;
      cooldownToggle.checked = perAgent > 0;
      if (cooldownField) cooldownField.hidden = !(perAgent > 0);
      if (cooldownInput) cooldownInput.value = perAgent > 0 ? perAgent : 0;
    }

    modelProvider?.addEventListener('change', (event) => {
      data.model.provider = event.target.value;
      applyProviderDefaults(data, 'model');
      persistSettings();
      renderAgents();
    });
    modelPreset?.addEventListener('blur', (event) => {
      data.model.preset = event.target.value;
      persistSettings();
    });
    modelEndpoint?.addEventListener('blur', (event) => {
      data.model.endpoint = event.target.value;
      persistSettings();
    });
    modelName?.addEventListener('change', (event) => {
      const value = event.target.value;
      if (isEmbeddingModel(value)) {
        event.target.setCustomValidity('Embedding models cannot be used for chat.');
        event.target.reportValidity();
        event.target.value = data.model.name ?? '';
        return;
      }
      event.target.setCustomValidity('');
      data.model.name = value;
      persistSettings();
    });
    modelName?.addEventListener('blur', (event) => {
      data.model.name = event.target.value;
      persistSettings();
    });
    modelKey?.addEventListener('blur', (event) => {
      data.model.key = event.target.value;
      persistSettings();
    });
    modelTemp?.addEventListener('change', (event) => {
      const value = Number(event.target.value);
      data.model.temp = Number.isFinite(value) ? value : 0.7;
      persistSettings();
    });
    modelMax?.addEventListener('change', (event) => {
      const value = Number(event.target.value);
      data.model.maxTokens = Number.isFinite(value) ? value : 2048;
      persistSettings();
    });

    embedProvider?.addEventListener('change', (event) => {
      data.embed.provider = event.target.value;
      applyProviderDefaults(data, 'embed');
      populateEmbeddingModels(embedModel, data.embed.model);
      persistSettings();
      renderAgents();
    });
    embedEndpoint?.addEventListener('blur', (event) => {
      data.embed.endpoint = event.target.value;
      persistSettings();
    });
    embedModel?.addEventListener('change', (event) => {
      data.embed.model = event.target.value;
      persistSettings();
    });
    embedKey?.addEventListener('blur', (event) => {
      data.embed.key = event.target.value;
      persistSettings();
    });
    embedCtx?.addEventListener('change', (event) => {
      const value = Number(event.target.value);
      data.embed.ctx = Number.isFinite(value) ? value : 2048;
      persistSettings();
    });

    persona?.addEventListener('blur', (event) => {
      data.persona = event.target.value;
      persistSettings();
    });

    removeBtn?.addEventListener('click', () => {
      if (Object.keys(settings.agents).length <= 1) {
        alert('At least one agent is required.');
        return;
      }
      delete settings.agents[id];
      persistSettings();
      renderAgents();
    });

    testChatBtn?.addEventListener('click', async () => {
      try {
        await runChatProbe(id, data);
        showToast(`Agent ${id} chat OK`);
      } catch (error) {
        console.error(error);
        showToast(`Agent ${id} chat failed: ${error.message}`, true);
      }
    });

    testEmbedBtn?.addEventListener('click', async () => {
      try {
        await runEmbedProbe(id, data);
        showToast(`Agent ${id} embedding OK`);
      } catch (error) {
        console.error(error);
        showToast(`Agent ${id} embedding failed: ${error.message}`, true);
      }
    });

    cooldownToggle?.addEventListener('change', (event) => {
      const enabled = Boolean(event.target.checked);
      if (cooldownField) cooldownField.hidden = !enabled;
      if (!enabled) {
        delete settings.arena.perAgentCooldown?.[id];
      } else {
        const value = Number(cooldownInput?.value ?? 0) || 0;
        settings.arena.perAgentCooldown = settings.arena.perAgentCooldown ?? {};
        settings.arena.perAgentCooldown[id] = value > 0 ? value : settings.arena.turnCooldownMs;
      }
      persistSettings();
    });

    cooldownInput?.addEventListener('change', (event) => {
      const value = Number(event.target.value);
      settings.arena.perAgentCooldown = settings.arena.perAgentCooldown ?? {};
      settings.arena.perAgentCooldown[id] = Number.isFinite(value) && value > 0 ? value : settings.arena.turnCooldownMs;
      persistSettings();
    });

    const accordionButton = fragment.querySelector('.accordion-toggle');
    const accordionBody = fragment.querySelector('.agent-card__body');
    if (accordionButton && accordionBody) {
      accordionButton.addEventListener('click', () => {
        const expanded = accordionButton.getAttribute('aria-expanded') === 'true';
        accordionButton.setAttribute('aria-expanded', String(!expanded));
        accordionBody.hidden = expanded;
      });
    }

    populateModelOptions(modelName, data.model.name);
    populateEmbeddingModels(embedModel, data.embed.model);

    elements.agentList.appendChild(fragment);
  }
  bindGlobalAccordions();
}

function bindGlobalAccordions() {
  if (accordionBound) return;
  accordionBound = true;
  document.addEventListener('click', (event) => {
    const button = event.target.closest('.accordion-toggle');
    if (!button) return;
    const container = button.closest('.agent-card');
    const body = container?.querySelector('.agent-card__body');
    if (!body) return;
    const expanded = button.getAttribute('aria-expanded') === 'true';
    button.setAttribute('aria-expanded', String(!expanded));
    body.hidden = expanded;
  });
}

function applyProviderDefaults(agent, scope) {
  const target = scope === 'embed' ? agent.embed : agent.model;
  const defaults = providerDefaults[target.provider];
  if (!defaults) return;
  if (scope === 'embed') {
    target.endpoint = defaults.embedEndpoint;
  } else {
    target.endpoint = defaults.chatEndpoint;
  }
}

async function runChatProbe(id, agent) {
  const persona = agent.persona?.trim();
  const messages = [];
  if (persona) {
    messages.push({ role: 'system', content: persona });
  }
  messages.push({ role: 'user', content: 'Ping from diagnostics. Reply with "pong".' });
  const response = await callChat(agent, messages);
  if (!response?.text?.toLowerCase().includes('pong')) {
    throw new Error('Unexpected response');
  }
}

async function runEmbedProbe(id, agent) {
  const result = await callEmbedding(agent, ['diagnostics ping']);
  if (!Array.isArray(result) || !Array.isArray(result[0])) {
    throw new Error('Invalid embedding response');
  }
}

function populateModelOptions(select, current) {
  if (!select) return;
  const options = new Set(modelCatalog.models);
  const snapshot = Array.from(options.values());
  select.innerHTML = '';
  const blank = document.createElement('option');
  blank.value = '';
  blank.textContent = 'Custom model';
  select.appendChild(blank);
  for (const name of snapshot) {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name;
    select.appendChild(option);
  }
  if (current && !options.has(current)) {
    const custom = document.createElement('option');
    custom.value = current;
    custom.textContent = `${current} (manual)`;
    select.appendChild(custom);
  }
  if (current) {
    select.value = current;
  }
}

function populateEmbeddingModels(select, current) {
  if (!select) return;
  const names = modelCatalog.embeds;
  select.innerHTML = '';
  const blank = document.createElement('option');
  blank.value = '';
  blank.textContent = 'None';
  select.appendChild(blank);
  for (const name of names) {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name;
    select.appendChild(option);
  }
  if (current && !names.includes(current)) {
    const custom = document.createElement('option');
    custom.value = current;
    custom.textContent = `${current} (manual)`;
    select.appendChild(custom);
  }
  if (current) select.value = current;
}

function nextAgentId() {
  const existing = Object.keys(settings.agents);
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  for (const char of alphabet) {
    if (!existing.includes(char)) {
      return char;
    }
  }
  return String(existing.length + 1);
}

function bindAgentAdder() {
  elements.addAgentBtn?.addEventListener('click', () => {
    const id = nextAgentId();
    settings.agents[id] = createAgent(id);
    persistSettings();
    renderAgents();
  });
}

function setupBoardControls() {
  elements.startArena?.addEventListener('click', () => {
    startArena();
  });
  elements.stopArena?.addEventListener('click', () => {
    stopArena();
  });
  elements.pauseArena?.addEventListener('click', () => {
    pauseArena();
  });
  elements.resumeArena?.addEventListener('click', () => {
    resumeArena();
  });
  elements.snapshotBoard?.addEventListener('click', async () => {
    try {
      await snapshotBoard();
      showToast('Snapshot saved.');
    } catch (error) {
      console.error(error);
      showToast(`Snapshot failed: ${error.message}`, true);
    }
  });
  elements.exportBoard?.addEventListener('click', () => {
    exportBoard();
  });
}

function setupHumanInput() {
  elements.humanToggle?.addEventListener('change', (event) => {
    const enabled = Boolean(event.target.checked);
    elements.chatPanel.hidden = !enabled;
  });
  elements.sendButton?.addEventListener('click', () => {
    submitHumanMessage();
  });
  elements.messageInput?.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      submitHumanMessage();
    }
  });
  elements.clearChatButton?.addEventListener('click', () => {
    elements.chatWindow.innerHTML = '';
  });
}

function submitHumanMessage() {
  const text = elements.messageInput?.value.trim();
  if (!text) return;
  elements.messageInput.value = '';
  const message = { role: 'human', content: text, ts: new Date().toISOString() };
  const bubble = document.createElement('div');
  bubble.className = 'board-entry';
  bubble.innerHTML = `<div class="board-entry__meta">Human • ${new Date(message.ts).toLocaleTimeString()}</div><div class="board-entry__content">${escapeHtml(text)}</div>`;
  elements.chatWindow?.appendChild(bubble);
  if (humanInputResolver) {
    humanInputResolver(message);
    humanInputResolver = null;
  } else {
    humanInputQueue.push(message);
  }
}

async function setupVectorCache() {
  if (unsubscribeVectorCache) {
    unsubscribeVectorCache();
    unsubscribeVectorCache = null;
  }
  unsubscribeVectorCache = subscribeVectorCache((stats) => {
    updateVectorCacheStatus(stats);
  });
  try {
    await initVectorCache();
  } catch (error) {
    console.warn('Vector cache initialisation failed', error);
  }
}

async function initialise() {
  populateReferences();
  bindOptionsHandle();
  loadSettings();
  renderArenaSettings();
  renderMemorySettings();
  bindArenaSettings();
  bindMemorySettings();
  bindAgentAdder();
  renderAgents();
  setupBoardControls();
  setupHumanInput();
  await setupVectorCache();
  await loadFloatingMemory();
  renderFloatingMemory();
  await loadBoard();
  renderBoard();
  scheduleOllamaRefresh();
}

async function loadBoard() {
  try {
    boardState = await boardStore.load();
  } catch (error) {
    console.error('Failed to load board state', error);
    boardState = deepClone(DEFAULT_BOARD);
  }
}

function renderBoard() {
  if (!elements.boardViewport) return;
  elements.boardViewport.innerHTML = '';
  for (const entry of boardState.entries) {
    const node = renderBoardEntry(entry);
    elements.boardViewport.appendChild(node);
  }
  updateBoardStatus();
}

function renderBoardEntry(entry) {
  const node = document.createElement('article');
  node.className = 'board-entry';
  const meta = document.createElement('div');
  meta.className = 'board-entry__meta';
  const label = formatAgentLabel(entry.agentId);
  const tokens = entry.tokens ? ` • ${entry.tokens} tokens` : '';
  meta.textContent = `Round ${entry.round} • Turn ${entry.turn} • ${label}${tokens}`;
  const content = document.createElement('div');
  content.className = 'board-entry__content';
  content.textContent = entry.content;
  node.append(meta, content);
  if (entry.summary) {
    const summary = document.createElement('div');
    summary.className = 'board-entry__summary';
    summary.textContent = entry.summary;
    node.appendChild(summary);
  }
  return node;
}

function updateBoardStatus() {
  if (!elements.boardStatus) return;
  const state = arenaState.running ? (arenaState.paused ? 'Paused' : 'Running') : 'Idle';
  elements.boardStatus.textContent = `Round ${arenaState.round} • ${state}`;
}

function appendBoardEntry(entry) {
  boardState.lastTurn = entry.turn;
  boardState.lastRound = entry.round;
  const node = renderBoardEntry(entry);
  elements.boardViewport?.appendChild(node);
  elements.boardViewport?.scrollTo({ top: elements.boardViewport.scrollHeight, behavior: 'smooth' });
  debouncedPersistBoard();
}

function debouncedPersistBoard() {
  if (pendingBoardPersist) {
    clearTimeout(pendingBoardPersist);
  }
  pendingBoardPersist = setTimeout(() => {
    pendingBoardPersist = null;
    void persistBoard();
  }, boardPersistDebounceMs);
}

async function persistBoard() {
  try {
    await boardStore.save(boardState);
  } catch (error) {
    console.error('Failed to persist board', error);
  }
}

async function persistArenaConfig() {
  if (typeof window === 'undefined') return;
  const fs = window.standaloneFs;
  if (!fs || typeof fs.writeJson !== 'function') {
    return;
  }
  try {
    const payload = {
      mode: settings.arena.mode,
      turnCooldownMs: settings.arena.turnCooldownMs,
      perAgentCooldown: settings.arena.perAgentCooldown,
      parallel: settings.arena.parallel,
      boardPolicy: settings.arena.boardPolicy,
      boardWindow: settings.arena.boardWindow,
      summaryEvery: settings.arena.summaryEvery,
      memory: settings.memory,
      agents: Object.fromEntries(
        Object.entries(settings.agents).map(([id, value]) => [
          id,
          {
            model: value.model,
            embed: value.embed,
            persona: value.persona
          }
        ])
      )
    };
    await fs.writeJson(ARENA_CONFIG_PATH, payload);
    for (const [id, agent] of Object.entries(settings.agents)) {
      await fs.writeJson(`${AGENT_CONFIG_DIR}/${id}.json`, agent);
    }
  } catch (error) {
    console.warn('Failed to persist arena configuration to disk', error);
  }
}

function scheduleOllamaRefresh() {
  refreshOllamaModels();
  if (ollamaRefreshTimer) {
    clearInterval(ollamaRefreshTimer);
  }
  ollamaRefreshTimer = setInterval(refreshOllamaModels, 60000);
}

async function refreshOllamaModels() {
  try {
    const response = await fetch(OLLAMA_TAGS_URL, { method: 'GET' });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    const models = Array.isArray(data?.models) ? data.models.map((item) => item.name).filter(Boolean) : [];
    const embedModels = models.filter((name) => OLLAMA_EMBED_REGEX.test(name));
    modelCatalog = { timestamp: Date.now(), models, embeds: embedModels };
    renderAgents();
  } catch (error) {
    console.warn('Failed to refresh Ollama models', error);
  }
}

function isEmbeddingModel(name) {
  return typeof name === 'string' && OLLAMA_EMBED_REGEX.test(name);
}

function buildBoardView(agentId) {
  const policy = settings.arena.boardPolicy;
  const windowSize = settings.arena.boardWindow;
  if (policy === 'last_k_messages') {
    return boardState.entries.slice(-windowSize);
  }
  if (policy === 'rolling_window_tokens') {
    const limit = windowSize;
    const selected = [];
    let total = 0;
    for (let index = boardState.entries.length - 1; index >= 0; index -= 1) {
      const entry = boardState.entries[index];
      const tokens = entry.tokens ?? estimateTokens(entry.content);
      if (total + tokens > limit && selected.length) {
        break;
      }
      selected.unshift(entry);
      total += tokens;
      if (total >= limit) break;
    }
    return selected;
  }
  // summary mode
  const recent = boardState.entries.slice(-windowSize);
  const summaries = boardState.entries.filter((entry) => entry.summary);
  const lastSummary = summaries[summaries.length - 1];
  const result = [];
  if (lastSummary) {
    result.push(lastSummary);
  }
  for (const entry of recent) {
    result.push(entry);
  }
  return result;
}

function formatAgentLabel(agentId) {
  if (agentId === humanAgentId) return 'Human';
  if (agentId === 'board') return 'Board';
  if (!agentId) return 'Agent';
  return `Agent ${agentId}`;
}

function estimateTokens(text) {
  if (!text) return 0;
  const words = text.trim().split(/\s+/);
  return Math.max(1, Math.ceil(words.length * 1.3));
}

async function gatherBoardContext(agentId, prompt) {
  const agent = settings.agents[agentId];
  if (!agent) return [];
  const slice = boardState.entries.slice(-Math.max(10, settings.arena.boardWindow));
  const originalById = new Map(slice.map((entry) => [entry.id, entry]));
  const candidates = applyBoardEmbeddings(boardEntriesToChunks(slice), slice.map((entry) => entry.embedding));
  if (!candidates.length) return [];
  let queryVector = null;
  try {
    const vectors = await maybeEmbed(agent, [prompt]);
    queryVector = vectors?.[0] ?? null;
  } catch (error) {
    console.warn('Failed to embed board query', error);
  }
  const results = await vectorSearch({
    candidates,
    limit: 3,
    useEmbeddings: Array.isArray(queryVector),
    queryVector,
    ensureEmbedding: async (entry) => {
      if (Array.isArray(entry.embedding) && entry.embedding.length) {
        return entry.embedding;
      }
      try {
        const vectors = await maybeEmbed(agent, [entry.content]);
        const vector = vectors?.[0] ?? null;
        entry.embedding = vector;
        const original = originalById.get(entry.id);
        if (original) original.embedding = vector;
        return vector;
      } catch (error) {
        console.warn('Failed to embed board entry', error);
        return null;
      }
    },
    lexicalScorer: (entry) => lexicalOverlap(prompt, entry.content),
    similarity: cosineSimilarity
  });
  return results.map((item) => item.entry);
}

async function retrieveAgentMemories(agentId, prompt) {
  const agent = settings.agents[agentId];
  if (!agent) return [];
  const cache = await loadAgentCache(agentId);
  const floating = floatingMemory.map((entry) => ({ ...entry, source: 'floating' }));
  const cached = Array.isArray(cache.entries)
    ? cache.entries.map((entry) => ({ ...entry, source: 'cache' }))
    : [];
  const candidates = [...cached, ...floating];
  if (!candidates.length) {
    return [];
  }
  let queryVector = null;
  if (settings.memory.useEmbeddings) {
    try {
      const vectors = await maybeEmbed(agent, [prompt]);
      queryVector = vectors?.[0] ?? null;
    } catch (error) {
      console.warn('Failed to embed agent query for memories', error);
    }
  }
  const limit = settings.memory.retrievalCount && settings.memory.retrievalCount > 0 ? settings.memory.retrievalCount : 5;
  const results = await vectorSearch({
    candidates,
    limit,
    useEmbeddings: Boolean(queryVector),
    queryVector,
    ensureEmbedding: async (entry) => {
      if (!settings.memory.useEmbeddings) return null;
      if (entry.source === 'cache') {
        if (Array.isArray(entry.embedding) && entry.embedding.length) {
          return entry.embedding;
        }
        try {
          const vectors = await maybeEmbed(agent, [entry.content]);
          const vector = vectors?.[0] ?? null;
          entry.embedding = vector;
          await persistAgentCache(agentId);
          return vector;
        } catch (error) {
          console.warn('Failed to embed cached memory entry', error);
          return null;
        }
      }
      if (entry.source === 'floating') {
        entry.embeddings = entry.embeddings ?? {};
        if (Array.isArray(entry.embeddings?.[agentId]) && entry.embeddings[agentId].length) {
          return entry.embeddings[agentId];
        }
        try {
          const vectors = await maybeEmbed(agent, [entry.content]);
          const vector = vectors?.[0] ?? null;
          if (!entry.embeddings) entry.embeddings = {};
          entry.embeddings[agentId] = vector;
          void persistFloatingMemory();
          return vector;
        } catch (error) {
          console.warn('Failed to embed floating memory entry', error);
          return null;
        }
      }
      return null;
    },
    lexicalScorer: (entry) => lexicalOverlap(prompt, entry.content),
    similarity: cosineSimilarity
  });
  return results.map((item) => item.entry);
}

function lexicalOverlap(a, b) {
  if (!a || !b) return 0;
  const tokensA = new Set(a.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean));
  const tokensB = new Set(b.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean));
  if (!tokensA.size || !tokensB.size) return 0;
  let overlap = 0;
  for (const token of tokensA) {
    if (tokensB.has(token)) overlap += 1;
  }
  return overlap / Math.max(tokensA.size, tokensB.size);
}

function cosineSimilarity(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || !a.length || !b.length || a.length !== b.length) {
    return 0;
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    const va = a[i] ?? 0;
    const vb = b[i] ?? 0;
    dot += va * vb;
    normA += va * va;
    normB += vb * vb;
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function maybeEmbed(agent, texts, options = {}) {
  const { force = false } = options;
  if (!Array.isArray(texts) || !texts.length) return null;
  if (!force && !settings.memory.useEmbeddings) {
    return null;
  }
  if (!agent) {
    return null;
  }
  const context = {
    agentId: resolveAgentId(agent),
    model: agent?.embed?.model || agent?.model?.name || agent?.model?.preset || 'default',
    provider: agent?.embed?.provider || 'generic'
  };
  try {
    await initVectorCache();
    const cached = await getCachedVectors({ ...context, texts });
    const results = cached.vectors.slice();
    const misses = cached.misses ?? [];
    if (!misses.length) {
      return results;
    }
    const inputs = misses.map((item) => item.text);
    const fresh = await callEmbedding(agent, inputs);
    if (!Array.isArray(fresh) || !fresh.length) {
      return results;
    }
    const storeEntries = [];
    for (let index = 0; index < misses.length; index += 1) {
      const miss = misses[index];
      const vector = fresh[index];
      if (!vector) continue;
      results[miss.index] = Array.isArray(vector) ? vector : Array.from(vector);
      storeEntries.push({ key: miss.key, hash: miss.hash, vector });
    }
    if (storeEntries.length) {
      await storeCachedVectors({ ...context, entries: storeEntries });
    }
    return results;
  } catch (error) {
    console.warn('Embedding request failed', error);
    return null;
  }
}

function buildPromptFromBoard(agentId) {
  const entries = buildBoardView(agentId);
  const persona = settings.agents[agentId]?.persona?.trim();
  let prompt = '';
  if (persona) {
    prompt += `You are ${persona}.\n`;
  }
  prompt += 'Shared bulletin board transcript:\n';
  for (const entry of entries) {
    const label = formatAgentLabel(entry.agentId);
    const prefix = entry.summary ? '[Summary]' : '';
    prompt += `${prefix}${label}: ${entry.content}\n`;
    if (entry.summary) {
      prompt += `Summary: ${entry.summary}\n`;
    }
  }
  prompt += '\nRespond with the next turn. Provide concise and relevant information.';
  return prompt;
}

function getAgentOrder(mode) {
  const ids = Object.keys(settings.agents).sort();
  if (mode === 'pair') {
    return ids.slice(0, 2);
  }
  if (mode === 'human') {
    return [humanAgentId, ...ids];
  }
  return ids;
}

async function startArena() {
  if (arenaState.running) {
    showToast('Arena already running.');
    return;
  }
  arenaState = createArenaState();
  arenaState.mode = settings.arena.mode;
  arenaState.order = getAgentOrder(settings.arena.mode);
  arenaState.running = true;
  arenaState.abortController = new AbortController();
  updateBoardStatus();
  if (arenaState.mode === 'allout' && settings.arena.parallel) {
    startParallelArena();
  } else {
    startSequentialArena();
  }
}

function pauseArena() {
  if (!arenaState.running) return;
  arenaState.paused = true;
  updateBoardStatus();
}

function resumeArena() {
  if (!arenaState.running) return;
  arenaState.paused = false;
  updateBoardStatus();
}

function stopArena() {
  arenaState.running = false;
  arenaState.paused = false;
  arenaState.abortController?.abort();
  arenaState.workers.forEach((worker) => worker.abort?.());
  arenaState.workers.clear();
  updateBoardStatus();
  void persistBoard();
}

async function startSequentialArena() {
  const { order } = arenaState;
  if (!order.length) {
    showToast('Add agents before starting the arena.', true);
    stopArena();
    return;
  }
  while (arenaState.running) {
    for (const id of order) {
      if (!arenaState.running) break;
      await waitWhilePaused();
      await maybeCooldown(id);
      if (id === humanAgentId) {
        await handleHumanTurn();
        continue;
      }
      const result = await runAgentTurn(id).catch((error) => {
        console.error('Agent turn failed', error);
        showToast(`Agent ${id} turn failed: ${error.message}`, true);
        return null;
      });
      if (!result) continue;
      recordBoardAppend(id, result);
    }
    arenaState.round += 1;
    updateBoardStatus();
  }
}

function startParallelArena() {
  const { order } = arenaState;
  if (!order.length) {
    showToast('Add agents before starting the arena.', true);
    stopArena();
    return;
  }
  for (const id of order) {
    if (id === humanAgentId) continue;
    const worker = createParallelWorker(id);
    arenaState.workers.set(id, worker);
    worker.loop();
  }
  if (order.includes(humanAgentId)) {
    (async () => {
      while (arenaState.running) {
        await waitWhilePaused();
        await maybeCooldown(humanAgentId);
        await handleHumanTurn();
      }
    })();
  }
}

function createParallelWorker(id) {
  const controller = new AbortController();
  return {
    abort: () => controller.abort(),
    async loop() {
      while (arenaState.running && !controller.signal.aborted) {
        await waitWhilePaused();
        await maybeCooldown(id);
        const result = await runAgentTurn(id).catch((error) => {
          console.error('Agent turn failed', error);
          showToast(`Agent ${id} turn failed: ${error.message}`, true);
          return null;
        });
        if (result) {
          recordBoardAppend(id, result);
        }
      }
    }
  };
}

async function waitWhilePaused() {
  while (arenaState.paused && arenaState.running) {
    await sleep(250);
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function maybeCooldown(id) {
  const perAgent = settings.arena.perAgentCooldown?.[id];
  const wait = Number.isFinite(perAgent) && perAgent > 0 ? perAgent : settings.arena.turnCooldownMs;
  if (wait > 0) {
    await sleep(wait);
  }
}

async function handleHumanTurn() {
  showToast('Waiting for human input…');
  const message = await waitForHumanMessage();
  const entry = {
    id: `board-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    turn: ++arenaState.turn,
    round: arenaState.round,
    agentId: humanAgentId,
    ts: new Date().toISOString(),
    tokens: estimateTokens(message.content),
    content: message.content
  };
  boardState.entries.push(entry);
  appendBoardEntry(entry);
  trackMemoryEntry(entry);
  updateBoardStatus();
}

function waitForHumanMessage() {
  if (humanInputQueue.length) {
    return Promise.resolve(humanInputQueue.shift());
  }
  return new Promise((resolve) => {
    humanInputResolver = resolve;
  });
}

async function runAgentTurn(id) {
  const entry = {
    id: `board-${Date.now()}-${id}`,
    round: arenaState.round,
    agentId: id,
    ts: new Date().toISOString()
  };
  const prompt = buildPromptFromBoard(id);
  const agent = settings.agents[id];
  const boardMemories = await gatherBoardContext(id, prompt);
  const agentMemories = await retrieveAgentMemories(id, prompt);
  arenaState.lastByAgent.set(id, { board: boardMemories, agent: agentMemories });
  let memoryNote = '';
  if (boardMemories.length) {
    memoryNote += `\nRelevant board memories:\n${boardMemories
      .map((record) => `${formatAgentLabel(record.agentId)}: ${record.content}`)
      .join('\n')}`;
  }
  if (settings.memory.autoInject && agentMemories.length) {
    memoryNote += `\nAgent memory hints:\n${agentMemories
      .map((record) => `${formatAgentLabel(record.agentId ?? id)}: ${record.content}`)
      .join('\n')}`;
  }
  const response = await callChat(agent, [{ role: 'user', content: prompt + memoryNote }]);
  entry.content = response.text;
  entry.tokens = response.tokens;
  arenaState.turn += 1;
  entry.turn = arenaState.turn;
  return entry;
}

function recordBoardAppend(id, entry) {
  boardState.entries.push(entry);
  boardState.lastTurn = entry.turn;
  boardState.lastRound = entry.round;
  if (arenaState.mode === 'allout' && settings.arena.parallel) {
    const participants = arenaState.order.filter((value) => value !== humanAgentId).length || 1;
    arenaState.round = Math.max(arenaState.round, Math.floor((arenaState.turn - 1) / participants) + 1);
  }
  appendBoardEntry(entry);
  if (!entry.summary) {
    trackMemoryEntry(entry);
    if (entry.agentId && entry.agentId !== humanAgentId && entry.agentId !== 'board') {
      void appendAgentCache(entry.agentId, entry);
    }
  }
  updateBoardStatus();
  maybeGenerateSummary(entry.turn);
}

async function maybeGenerateSummary(turn) {
  if (settings.arena.boardPolicy !== 'summary_every_n_turns') {
    return;
  }
  if (turn % settings.arena.summaryEvery !== 0) {
    return;
  }
  const windowEntries = buildBoardView(null);
  const summaryText = await synthesizeSummary(windowEntries).catch((error) => {
    console.warn('Summary synthesis failed', error);
    return null;
  });
  if (!summaryText) return;
  const summaryEntry = {
    id: `summary-${Date.now()}`,
    round: arenaState.round,
    turn,
    agentId: 'board',
    ts: new Date().toISOString(),
    content: 'Board summary',
    summary: summaryText,
    tokens: estimateTokens(summaryText)
  };
  boardState.entries.push(summaryEntry);
  appendBoardEntry(summaryEntry);
}

async function synthesizeSummary(entries) {
  const text = entries
    .map((entry) => `${formatAgentLabel(entry.agentId)}: ${entry.content}`)
    .join('\n');
  if (!text) return null;
  for (const [id, agent] of Object.entries(settings.agents)) {
    if (!agent.model.name) continue;
    try {
      const response = await callChat(agent, [
        { role: 'system', content: 'Summarize the bulletin board segment succinctly.' },
        { role: 'user', content: text }
      ]);
      if (response?.text) {
        return response.text;
      }
    } catch (error) {
      console.warn(`Summary attempt with agent ${id} failed`, error);
      continue;
    }
  }
  return text.slice(0, 400) + (text.length > 400 ? '…' : '');
}

async function snapshotBoard() {
  const windowEntries = buildBoardView(null);
  const summary = await synthesizeSummary(windowEntries);
  if (!summary) throw new Error('Nothing to summarize');
  const summaryEntry = {
    id: `snapshot-${Date.now()}`,
    round: arenaState.round,
    turn: arenaState.turn,
    agentId: 'board',
    ts: new Date().toISOString(),
    content: 'Manual snapshot',
    summary,
    tokens: estimateTokens(summary)
  };
  boardState.entries.push(summaryEntry);
  appendBoardEntry(summaryEntry);
  await persistBoard();
}

function exportBoard() {
  const rows = boardState.entries.map((entry) => {
    const label = entry.agentId === humanAgentId ? 'Human' : entry.agentId;
    return `${entry.round}\t${entry.turn}\t${label}\t${entry.content.replace(/\n/g, ' ')}`;
  });
  const blob = new Blob([rows.join('\n')], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `sam-board-${Date.now()}.txt`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

async function callChat(agent, messages) {
  if (!agent || !agent.model) {
    throw new Error('Agent configuration missing.');
  }
  if (isEmbeddingModel(agent.model.name)) {
    throw new Error('Embedding model selected for chat. Choose a chat-capable model.');
  }
  const provider = agent.model.provider ?? 'ollama_local';
  if (!chatProviders.has(provider)) {
    throw new Error(`Unsupported chat provider: ${provider}`);
  }
  const endpoint = agent.model.endpoint || providerDefaults[provider]?.chatEndpoint;
  if (!endpoint) {
    throw new Error('Chat endpoint missing.');
  }
  if (!agent.model.name && !agent.model.preset) {
    throw new Error('Select a chat model before starting the arena.');
  }
  const payload = buildChatPayload(provider, agent, messages);
  const start = performance.now();
  const response = await safeFetch(endpoint, {
    method: 'POST',
    headers: buildHeaders(agent.model.key, provider),
    body: JSON.stringify(payload)
  });
  const duration = performance.now() - start;
  console.info(`[Agent=${agent.id ?? 'unknown'}] [Provider=${provider}] [CHAT] URL=${endpoint} t=${Math.round(duration)}ms`);
  if (provider === 'ollama_local') {
    const data = await response.json();
    const text = data.response ?? data.generated_text ?? '';
    const tokens = data.eval_count ?? estimateTokens(text);
    return { text, tokens };
  }
  const data = await response.json();
  const choice = data.choices?.[0];
  const text = choice?.message?.content ?? choice?.text ?? '';
  const tokens = data.usage?.total_tokens ?? estimateTokens(text);
  return { text, tokens };
}

function buildChatPayload(provider, agent, messages) {
  if (provider === 'ollama_local') {
    const flattened = messages.map((message) => message.content).join('\n');
    return {
      model: agent.model.name || agent.model.preset,
      prompt: flattened,
      options: {
        temperature: agent.model.temp ?? 0.7,
        num_predict: agent.model.maxTokens ?? 2048
      },
      stream: false
    };
  }
  const openAiMessages = messages.map((message) => ({
    role: message.role === 'assistant' ? 'assistant' : message.role === 'system' ? 'system' : 'user',
    content: message.content
  }));
  return {
    model: agent.model.name || agent.model.preset || 'gpt-3.5-turbo',
    messages: openAiMessages,
    temperature: agent.model.temp ?? 0.7,
    max_tokens: agent.model.maxTokens ?? 2048,
    stream: false
  };
}

async function callEmbedding(agent, inputs) {
  if (!agent || !agent.embed) {
    throw new Error('Embedding configuration missing');
  }
  const provider = agent.embed.provider ?? 'ollama_local';
  const endpoint = agent.embed.endpoint || providerDefaults[provider]?.embedEndpoint;
  if (!endpoint) {
    throw new Error('Embedding endpoint missing');
  }
  const payload = buildEmbedPayload(provider, agent, inputs);
  if (!payload.model) {
    throw new Error('Select an embedding model before running retrieval.');
  }
  const start = performance.now();
  const response = await safeFetch(endpoint, {
    method: 'POST',
    headers: buildHeaders(agent.embed.key, provider),
    body: JSON.stringify(payload)
  }, embedTimeoutMs);
  const duration = performance.now() - start;
  console.info(`[Agent=${agent.id ?? 'unknown'}] [Provider=${provider}] [EMBED] URL=${endpoint} t=${Math.round(duration)}ms`);
  if (provider === 'ollama_local') {
    const data = await response.json();
    if (Array.isArray(data.embeddings)) {
      return data.embeddings;
    }
    return data.embedding ? [data.embedding] : [];
  }
  const data = await response.json();
  const vectors = Array.isArray(data.data) ? data.data.map((item) => item.embedding ?? []) : [];
  return vectors;
}

function buildEmbedPayload(provider, agent, inputs) {
  if (provider === 'ollama_local') {
    return {
      model: agent.embed.model || agent.model.name,
      input: inputs
    };
  }
  return {
    model: agent.embed.model || agent.model.name || 'text-embedding-3-small',
    input: inputs
  };
}

async function safeFetch(url, options, timeout = chatTimeoutMs) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    if (!response.ok) {
      if ((response.status === 404 || response.status === 405) && options?.method === 'POST') {
        const remapped = remapEndpoint(url);
        if (remapped !== url) {
          showToast(`[Bridge] Remapped to ${remapped}`);
          return safeFetch(remapped, options, timeout);
        }
      }
      throw new Error(`HTTP ${response.status}`);
    }
    return response;
  } finally {
    clearTimeout(id);
  }
}

function remapEndpoint(url) {
  if (!url) return url;
  if (url.includes('/v1/chat') && url.startsWith('http://localhost:11434')) {
    return 'http://localhost:11434/api/generate';
  }
  if (url.includes('/v1/embeddings') && url.startsWith('http://localhost:11434')) {
    return 'http://localhost:11434/api/embed';
  }
  return url;
}

function buildHeaders(key, provider) {
  const headers = { 'Content-Type': 'application/json' };
  if (key) {
    headers.Authorization = `Bearer ${key}`;
  }
  if (provider === 'ollama_local') {
    delete headers.Authorization;
  }
  return headers;
}

function showToast(message, isError = false) {
  console[isError ? 'error' : 'info'](message);
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (match) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  })[match]);
}

if (document.readyState === 'complete' || document.readyState === 'interactive') {
  void initialise();
} else {
  document.addEventListener('DOMContentLoaded', () => {
    void initialise();
  });
}

