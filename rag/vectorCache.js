const MB = 1024 * 1024;
const STORE_NAME = 'vectors';
const INDEX_NAME = 'lastAccess';
const DB_VERSION = 1;

const hasIndexedDb = typeof indexedDB !== 'undefined';

const state = {
  db: null,
  dbName: 'sam-vectors',
  maxBytes: 100 * MB,
  entries: 0,
  bytes: 0,
  statsDirty: true
};

function normaliseVector(vector) {
  if (!vector) return null;
  let array;
  if (vector instanceof Float32Array) {
    array = vector;
  } else if (Array.isArray(vector) || ArrayBuffer.isView(vector)) {
    array = Float32Array.from(vector);
  } else {
    return null;
  }
  let norm = 0;
  for (let i = 0; i < array.length; i += 1) {
    const value = array[i] ?? 0;
    norm += value * value;
  }
  if (norm === 0) {
    return array;
  }
  const length = Math.sqrt(norm);
  if (!Number.isFinite(length) || length === 0) {
    return array;
  }
  const scaled = new Float32Array(array.length);
  const factor = 1 / length;
  for (let i = 0; i < array.length; i += 1) {
    scaled[i] = (array[i] ?? 0) * factor;
  }
  return scaled;
}

async function ensureOpen(options = {}) {
  if (!hasIndexedDb) {
    return null;
  }
  if (state.db) {
    return state.db;
  }
  state.dbName = options.dbName || state.dbName;
  state.maxBytes = Number.isFinite(options.maxMB) ? options.maxMB * MB : state.maxBytes;
  state.db = await new Promise((resolve, reject) => {
    const request = indexedDB.open(state.dbName, DB_VERSION);
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
        store.createIndex(INDEX_NAME, 'lastAccess', { unique: false });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  console.debug('[VectorCache] open → ok');
  state.statsDirty = true;
  await computeStats();
  await evictIfNeeded();
  return state.db;
}

async function computeStats() {
  if (!hasIndexedDb || !state.db) {
    state.entries = 0;
    state.bytes = 0;
    state.statsDirty = false;
    return { entries: 0, bytes: 0 };
  }
  const tx = state.db.transaction(STORE_NAME, 'readonly');
  const store = tx.objectStore(STORE_NAME);
  let entries = 0;
  let bytes = 0;
  await new Promise((resolve, reject) => {
    const request = store.openCursor();
    request.onsuccess = (event) => {
      const cursor = event.target.result;
      if (!cursor) {
        resolve();
        return;
      }
      const value = cursor.value;
      entries += 1;
      bytes += Number(value.bytes) || 0;
      cursor.continue();
    };
    request.onerror = () => reject(request.error);
  });
  state.entries = entries;
  state.bytes = bytes;
  state.statsDirty = false;
  return { entries, bytes };
}

async function getSize() {
  if (state.statsDirty) {
    await computeStats();
  }
  return { entries: state.entries, bytes: state.bytes };
}

async function get(queryEmbedding, options = {}) {
  const { topK = 12, minSim = 0.18 } = options;
  if (!hasIndexedDb || !state.db) {
    return [];
  }
  const query = normaliseVector(queryEmbedding);
  if (!query || !query.length) {
    return [];
  }
  const matches = [];
  const now = Date.now();
  const tx = state.db.transaction(STORE_NAME, 'readwrite');
  const store = tx.objectStore(STORE_NAME);
  await new Promise((resolve, reject) => {
    const request = store.openCursor();
    request.onsuccess = (event) => {
      const cursor = event.target.result;
      if (!cursor) {
        resolve();
        return;
      }
      const value = cursor.value;
      const embedding = new Float32Array(value.emb);
      let dot = 0;
      for (let i = 0; i < query.length && i < embedding.length; i += 1) {
        dot += (query[i] ?? 0) * (embedding[i] ?? 0);
      }
      const similarity = dot;
      if (similarity >= minSim) {
        matches.push({
          id: value.id,
          embedding,
          meta: value.meta || null,
          similarity
        });
        matches.sort((a, b) => b.similarity - a.similarity);
        if (matches.length > topK) {
          matches.length = topK;
        }
      }
      value.lastAccess = now;
      cursor.update(value);
      cursor.continue();
    };
    request.onerror = () => reject(request.error);
  });
  const hits = matches.length;
  const misses = Math.max(0, topK - hits);
  console.debug(`[VectorCache] hits: ${hits}, misses: ${misses}, topK: ${topK}`);
  return matches;
}

async function putMany(entries) {
  if (!hasIndexedDb || !state.db) {
    return;
  }
  if (!Array.isArray(entries) || !entries.length) {
    return;
  }
  const tx = state.db.transaction(STORE_NAME, 'readwrite');
  const store = tx.objectStore(STORE_NAME);
  await new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error || new Error('Vector cache transaction aborted'));
    for (const entry of entries) {
      if (!entry || !entry.id || !entry.embedding) {
        continue;
      }
      const vector = normaliseVector(entry.embedding);
      if (!vector) {
        continue;
      }
      const record = {
        id: entry.id,
        emb: vector.buffer.slice(0),
        dim: vector.length,
        meta: entry.meta || null,
        lastAccess: Date.now(),
        bytes: vector.byteLength
      };
      store.put(record);
    }
  });
  state.statsDirty = true;
  await evictIfNeeded();
}

async function evictIfNeeded() {
  if (!hasIndexedDb || !state.db) {
    return;
  }
  if (state.statsDirty) {
    await computeStats();
  }
  if (state.bytes <= state.maxBytes) {
    return;
  }
  const beforeMb = state.bytes / MB;
  const tx = state.db.transaction(STORE_NAME, 'readwrite');
  const store = tx.objectStore(STORE_NAME);
  const index = store.index(INDEX_NAME);
  let removed = 0;
  await new Promise((resolve, reject) => {
    const request = index.openCursor();
    request.onsuccess = (event) => {
      const cursor = event.target.result;
      if (!cursor) {
        resolve();
        return;
      }
      if (state.bytes <= state.maxBytes) {
        resolve();
        return;
      }
      const value = cursor.value;
      state.bytes = Math.max(0, state.bytes - (Number(value.bytes) || 0));
      state.entries = Math.max(0, state.entries - 1);
      removed += 1;
      cursor.delete();
      cursor.continue();
    };
    request.onerror = () => reject(request.error);
  });
  state.statsDirty = true;
  await computeStats();
  const afterMb = state.bytes / MB;
  console.debug(
    `[VectorCache] evict: bytes→MB before=${beforeMb.toFixed(2)} after=${afterMb.toFixed(2)}, removed: ${removed}`
  );
}

export const VectorCache = {
  async open(options = {}) {
    if (!hasIndexedDb) {
      return { entries: 0, bytes: 0, supported: false };
    }
    const db = await ensureOpen(options);
    if (!db) {
      return { entries: 0, bytes: 0, supported: false };
    }
    const size = await getSize();
    return { ...size, supported: true };
  },

  async get(queryEmbedding, options = {}) {
    return get(queryEmbedding, options);
  },

  async putMany(entries) {
    await putMany(entries);
  },

  async size() {
    const size = await getSize();
    return { ...size, supported: hasIndexedDb };
  },

  async evictIfNeeded() {
    await evictIfNeeded();
  }
};

export default VectorCache;
