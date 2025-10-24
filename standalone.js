const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');

const STORAGE_DIR_NAME = 'sam-storage';
const ramStore = new Map();
const APP_BASE_PATH = path.resolve(__dirname);
const BINARY_EXTENSIONS = new Set(['.pdf']);

function normalizeRelativePath(input) {
  if (!input) return '';
  const normalized = String(input).replace(/\\/g, '/');
  return normalized.replace(/^\/+/, '');
}

function resolveAppPath(relativePath) {
  const normalized = normalizeRelativePath(relativePath);
  const target = path.resolve(APP_BASE_PATH, normalized || '.');
  if (!target.startsWith(APP_BASE_PATH)) {
    throw new Error(`Refusing to access path outside app directory: ${relativePath}`);
  }
  return target;
}

function inferContentTypeFromPath(relativePath) {
  const extension = path.extname(relativePath || '').replace('.', '').toLowerCase();
  if (extension === 'json' || extension === 'jsonl') {
    return 'application/json';
  }
  if (extension === 'pdf') {
    return 'application/pdf';
  }
  if (extension === 'md' || extension === 'markdown' || extension === 'txt' || extension === 'text') {
    return 'text/plain';
  }
  return 'text/plain';
}

function ensureAppDirectory(relativeDir) {
  const normalized = normalizeRelativePath(relativeDir);
  const target = resolveAppPath(normalized || '.');
  if (!fs.existsSync(target)) {
    fs.mkdirSync(target, { recursive: true });
  }
  return true;
}

function listAppDirectory(relativeDir) {
  const normalized = normalizeRelativePath(relativeDir);
  ensureAppDirectory(normalized);
  const target = resolveAppPath(normalized || '.');
  const entries = fs.readdirSync(target, { withFileTypes: true });
  const prefix = normalized ? normalized.replace(/\/+$/, '') : '';
  const results = [];
  for (const entry of entries) {
    if (!entry.isFile()) continue;
    const entryPath = prefix ? `${prefix}/${entry.name}` : entry.name;
    results.push(entryPath);
  }
  return results;
}

function readAppFile(relativePath) {
  const normalized = normalizeRelativePath(relativePath);
  if (!normalized) return null;
  const target = resolveAppPath(normalized);
  if (!fs.existsSync(target)) {
    return null;
  }
  const stats = fs.lstatSync(target);
  if (stats.isDirectory()) {
    return null;
  }
  const extension = path.extname(normalized).toLowerCase();
  if (BINARY_EXTENSIONS.has(extension)) {
    return null;
  }
  const content = fs.readFileSync(target, 'utf8');
  return { content, contentType: inferContentTypeFromPath(normalized) };
}

function writeAppFile(relativePath, content) {
  const normalized = normalizeRelativePath(relativePath);
  if (!normalized) return false;
  const extension = path.extname(normalized).toLowerCase();
  if (BINARY_EXTENSIONS.has(extension)) {
    return false;
  }
  const dirName = path.posix.dirname(normalized);
  if (dirName && dirName !== '.') {
    ensureAppDirectory(dirName);
  }
  const target = resolveAppPath(normalized);
  fs.writeFileSync(target, String(content ?? ''), 'utf8');
  return true;
}

function deleteAppFile(relativePath) {
  const normalized = normalizeRelativePath(relativePath);
  if (!normalized) return false;
  const target = resolveAppPath(normalized);
  if (!fs.existsSync(target)) {
    return true;
  }
  const stats = fs.lstatSync(target);
  if (!stats.isFile()) {
    return false;
  }
  fs.unlinkSync(target);
  return true;
}

function ensureStorageDir() {
  const dir = path.join(app.getPath('userData'), STORAGE_DIR_NAME);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  return dir;
}

function encodeKey(key) {
  return Buffer.from(String(key)).toString('base64url');
}

function decodeKey(filename) {
  try {
    return Buffer.from(filename, 'base64url').toString('utf8');
  } catch (error) {
    console.warn('Failed to decode storage key', filename, error);
    return null;
  }
}

function storagePathForKey(key) {
  const dir = ensureStorageDir();
  const encoded = encodeKey(key);
  return path.join(dir, encoded);
}

function readStorageValue(key) {
  if (ramStore.has(key)) {
    return ramStore.get(key);
  }
  try {
    const filePath = storagePathForKey(key);
    if (!fs.existsSync(filePath)) {
      return null;
    }
    const value = fs.readFileSync(filePath, 'utf8');
    ramStore.set(key, value);
    return value;
  } catch (error) {
    console.warn('Failed to read storage value for key', key, error);
    return null;
  }
}

function writeStorageValue(key, value) {
  try {
    const filePath = storagePathForKey(key);
    const normalized = String(value);
    ramStore.set(key, normalized);
    fs.writeFileSync(filePath, normalized, 'utf8');
    return true;
  } catch (error) {
    console.warn('Failed to persist storage value for key', key, error);
    ramStore.set(key, String(value));
    return false;
  }
}

function removeStorageValue(key) {
  try {
    ramStore.delete(key);
    const filePath = storagePathForKey(key);
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
    return true;
  } catch (error) {
    console.warn('Failed to remove storage value for key', key, error);
    return false;
  }
}

function listStorageKeys() {
  if (ramStore.size > 0) {
    return Array.from(ramStore.keys());
  }
  try {
    const dir = ensureStorageDir();
    const entries = fs.readdirSync(dir);
    const keys = [];
    for (const entry of entries) {
      const key = decodeKey(entry);
      if (key) {
        try {
          const value = fs.readFileSync(path.join(dir, entry), 'utf8');
          ramStore.set(key, value);
          keys.push(key);
        } catch (error) {
          console.warn('Failed to hydrate RAM store for key', key, error);
        }
      }
    }
    return keys;
  } catch (error) {
    console.warn('Failed to list storage keys', error);
    return Array.from(ramStore.keys());
  }
}

function clearStorage() {
  try {
    ramStore.clear();
    const dir = ensureStorageDir();
    if (!fs.existsSync(dir)) {
      return true;
    }
    const entries = fs.readdirSync(dir);
    for (const entry of entries) {
      const target = path.join(dir, entry);
      try {
        if (fs.lstatSync(target).isFile()) {
          fs.unlinkSync(target);
        }
      } catch (error) {
        console.warn('Failed to remove storage file', target, error);
      }
    }
    return true;
  } catch (error) {
    console.warn('Failed to clear storage', error);
    return false;
  }
}

function primeRamStore() {
  try {
    const dir = ensureStorageDir();
    if (!fs.existsSync(dir)) {
      return;
    }
    const entries = fs.readdirSync(dir);
    for (const entry of entries) {
      const key = decodeKey(entry);
      if (!key) continue;
      const filePath = path.join(dir, entry);
      try {
        const value = fs.readFileSync(filePath, 'utf8');
        ramStore.set(key, value);
      } catch (error) {
        console.warn('Failed to preload RAM store value for key', key, error);
      }
    }
  } catch (error) {
    console.warn('Failed to prime RAM store', error);
  }
}

function registerStorageHandlers() {
  ipcMain.on('standalone-store-get', (event, key) => {
    event.returnValue = readStorageValue(key);
  });

  ipcMain.on('standalone-store-set', (event, key, value) => {
    event.returnValue = writeStorageValue(key, value);
  });

  ipcMain.on('standalone-store-remove', (event, key) => {
    event.returnValue = removeStorageValue(key);
  });

  ipcMain.on('standalone-store-keys', (event) => {
    event.returnValue = listStorageKeys();
  });

  ipcMain.on('standalone-store-clear', (event) => {
    event.returnValue = clearStorage();
  });
}

function registerFilesystemHandlers() {
  ipcMain.on('standalone-fs-list', (event, relativeDir) => {
    try {
      event.returnValue = listAppDirectory(relativeDir);
    } catch (error) {
      console.warn('Failed to list directory for renderer', relativeDir, error);
      event.returnValue = [];
    }
  });

  ipcMain.on('standalone-fs-read', (event, relativePath) => {
    try {
      event.returnValue = readAppFile(relativePath);
    } catch (error) {
      console.warn('Failed to read file for renderer', relativePath, error);
      event.returnValue = null;
    }
  });

  ipcMain.on('standalone-fs-write', (event, relativePath, content) => {
    try {
      event.returnValue = writeAppFile(relativePath, content);
    } catch (error) {
      console.warn('Failed to write file for renderer', relativePath, error);
      event.returnValue = false;
    }
  });

  ipcMain.on('standalone-fs-delete', (event, relativePath) => {
    try {
      event.returnValue = deleteAppFile(relativePath);
    } catch (error) {
      console.warn('Failed to delete file for renderer', relativePath, error);
      event.returnValue = false;
    }
  });

  ipcMain.on('standalone-fs-ensure', (event, relativeDir) => {
    try {
      event.returnValue = ensureAppDirectory(relativeDir);
    } catch (error) {
      console.warn('Failed to ensure directory for renderer', relativeDir, error);
      event.returnValue = false;
    }
  });
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1280,
    height: 840,
    webPreferences: {
      preload: path.join(__dirname, 'standalone-preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  });

  win.loadFile(path.join(__dirname, 'index.html'));

  if (process.env.STANDALONE_DEVTOOLS === '1') {
    win.webContents.openDevTools({ mode: 'detach' });
  }
}

app.whenReady().then(() => {
  ensureStorageDir();
  primeRamStore();
  registerStorageHandlers();
  registerFilesystemHandlers();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
