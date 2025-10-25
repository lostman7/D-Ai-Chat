const DB_NAME = 'sam_vector_cache';
const STORE_NAME = 'vectors';
const META_STORE = 'meta';
const MAX_CACHE_BYTES = 100 * 1024 * 1024; // 100 MB

const hasIndexedDb = typeof indexedDB !== 'undefined';

let dbPromise = null;
let statsLoaded = false;
let pendingOperations = 0;
const listeners = new Set();

const cacheStats = {
  entries: 0,
  bytes: 0,
  status: hasIndexedDb ? 'Idle' : 'Unavailable',
  supported: hasIndexedDb
};

function notify() {
  const snapshot = { ...cacheStats };
  for (const listener of listeners) {
    try {
      listener(snapshot);
    } catch (error) {
      console.error('Vector cache listener failed', error);
    }
  }
}

function beginOperation(status) {
  if (!hasIndexedDb) return;
  pendingOperations += 1;
  cacheStats.status = status;
  notify();
}

function endOperation() {
  if (!hasIndexedDb) return;
  pendingOperations = Math.max(0, pendingOperations - 1);
  if (pendingOperations === 0) {
    cacheStats.status = 'Idle';
    notify();
  }
}

function simpleHash(input) {
  if (!input) return '0';
  let hash = 2166136261;
  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16);
}

function buildCacheKey({ agentId, model, provider, textHash }) {
  const agent = agentId || 'unknown';
  const modelId = model || 'default';
  const providerId = provider || 'generic';
  return `${providerId}::${modelId}::${agent}::${textHash}`;
}

function toFloat32Array(vector) {
  if (!vector) return null;
  if (vector instanceof Float32Array) return vector;
  if (Array.isArray(vector)) return Float32Array.from(vector);
  if (ArrayBuffer.isView(vector)) return Float32Array.from(vector);
  return null;
}

function requestToPromise(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function openDb() {
  if (!hasIndexedDb) return null;
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'key' });
        store.createIndex('updatedAt', 'updatedAt', { unique: false });
      }
      if (!db.objectStoreNames.contains(META_STORE)) {
        db.createObjectStore(META_STORE, { keyPath: 'id' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  return dbPromise;
}

async function syncStats(db) {
  if (!hasIndexedDb || !db) return;
  try {
    const tx = db.transaction(META_STORE, 'readonly');
    const store = tx.objectStore(META_STORE);
    const record = await requestToPromise(store.get('stats'));
    cacheStats.entries = Number.isFinite(record?.entries) ? record.entries : 0;
    cacheStats.bytes = Number.isFinite(record?.bytes) ? record.bytes : 0;
    statsLoaded = true;
    notify();
  } catch (error) {
    console.warn('Vector cache stats sync failed', error);
  }
}

async function ensureStats(db) {
  if (!statsLoaded && db) {
    await syncStats(db);
  }
}

async function evictIfNeeded(db) {
  if (!hasIndexedDb || !db) return;
  if (cacheStats.bytes <= MAX_CACHE_BYTES) return;
  beginOperation('Syncing…');
  try {
    const tx = db.transaction([STORE_NAME, META_STORE], 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const meta = tx.objectStore(META_STORE);
    await new Promise((resolve, reject) => {
      const index = store.index('updatedAt');
      const request = index.openCursor();
      request.onsuccess = (event) => {
        const cursor = event.target.result;
        if (!cursor) {
          resolve();
          return;
        }
        if (cacheStats.bytes <= MAX_CACHE_BYTES) {
          resolve();
          return;
        }
        const record = cursor.value;
        cacheStats.bytes = Math.max(0, cacheStats.bytes - (record.size ?? 0));
        cacheStats.entries = Math.max(0, cacheStats.entries - 1);
        cursor.delete();
        cursor.continue();
      };
      request.onerror = (event) => reject(event.target.error);
    });
    await requestToPromise(meta.put({ id: 'stats', bytes: cacheStats.bytes, entries: cacheStats.entries }));
  } catch (error) {
    console.warn('Vector cache eviction failed', error);
  } finally {
    endOperation();
    notify();
  }
}

export function subscribeVectorCache(listener) {
  if (typeof listener !== 'function') {
    return () => {};
  }
  listeners.add(listener);
  listener({ ...cacheStats });
  return () => {
    listeners.delete(listener);
  };
}

export async function initVectorCache() {
  if (!hasIndexedDb) {
    cacheStats.status = 'Unavailable';
    notify();
    return { ...cacheStats };
  }
  try {
    const db = await openDb();
    await syncStats(db);
  } catch (error) {
    console.warn('Vector cache initialisation failed', error);
    cacheStats.status = 'Error';
    notify();
  }
  return { ...cacheStats };
}

export async function getCachedVectors({ agentId, model, provider, texts }) {
  const result = {
    vectors: Array.isArray(texts) ? new Array(texts.length).fill(null) : [],
    misses: []
  };
  if (!Array.isArray(texts) || !texts.length) {
    return result;
  }
  if (!hasIndexedDb) {
    result.misses = texts.map((text, index) => ({
      index,
      text,
      hash: simpleHash(text),
      key: null
    }));
    return result;
  }
  const db = await openDb();
  await ensureStats(db);
  beginOperation('Hydrating…');
  try {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    await Promise.all(texts.map((text, index) => {
      const hash = simpleHash(text);
      const key = buildCacheKey({ agentId, model, provider, textHash: hash });
      return new Promise((resolve) => {
        const request = store.get(key);
        request.onsuccess = (event) => {
          const record = event.target.result;
          if (record && record.vector) {
            console.debug(`[VectorCache] Hit ${key}`);
            result.vectors[index] = Array.from(record.vector);
          } else {
            console.debug(`[VectorCache] Miss ${key}`);
            result.misses.push({ index, text, hash, key });
          }
          resolve();
        };
        request.onerror = () => {
          console.debug(`[VectorCache] Miss ${key} (error)`);
          result.misses.push({ index, text, hash, key });
          resolve();
        };
      });
    }));
  } finally {
    endOperation();
  }
  return result;
}

export async function storeCachedVectors({ agentId, model, provider, entries }) {
  if (!hasIndexedDb || !Array.isArray(entries) || !entries.length) {
    return;
  }
  const db = await openDb();
  await ensureStats(db);
  beginOperation('Syncing…');
  try {
    const tx = db.transaction([STORE_NAME, META_STORE], 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const meta = tx.objectStore(META_STORE);
    let totalBytes = cacheStats.bytes;
    let totalEntries = cacheStats.entries;
    for (const entry of entries) {
      if (!entry || !entry.vector) continue;
      const hash = entry.hash || simpleHash(entry.text || '');
      const key = entry.key || buildCacheKey({ agentId, model, provider, textHash: hash });
      const vector = toFloat32Array(entry.vector);
      if (!vector) continue;
      const size = vector.byteLength;
      const existing = await requestToPromise(store.get(key)).catch(() => null);
      const record = {
        key,
        agentId,
        model,
        provider,
        textHash: hash,
        size,
        vector,
        createdAt: existing?.createdAt ?? Date.now(),
        updatedAt: Date.now()
      };
      await requestToPromise(store.put(record));
      totalBytes += size - (existing?.size ?? 0);
      if (!existing) {
        totalEntries += 1;
      }
    }
    cacheStats.bytes = totalBytes;
    cacheStats.entries = totalEntries;
    await requestToPromise(meta.put({ id: 'stats', bytes: totalBytes, entries: totalEntries }));
  } catch (error) {
    console.warn('Vector cache store failed', error);
  } finally {
    endOperation();
    notify();
  }
  await evictIfNeeded(db);
}

export const VECTOR_CACHE_LIMIT_BYTES = MAX_CACHE_BYTES;
