const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('standaloneStore', {
  getItem(key) {
    return ipcRenderer.sendSync('standalone-store-get', key);
  },
  setItem(key, value) {
    return ipcRenderer.sendSync('standalone-store-set', key, value);
  },
  removeItem(key) {
    return ipcRenderer.sendSync('standalone-store-remove', key);
  },
  keys() {
    return ipcRenderer.sendSync('standalone-store-keys');
  },
  clear() {
    return ipcRenderer.sendSync('standalone-store-clear');
  }
});

contextBridge.exposeInMainWorld('standaloneFs', {
  listDirectory(target) {
    return ipcRenderer.sendSync('standalone-fs-list', target);
  },
  readFile(target) {
    return ipcRenderer.sendSync('standalone-fs-read', target);
  },
  writeFile(target, content, options) {
    return ipcRenderer.sendSync('standalone-fs-write', target, content, options);
  },
  deleteFile(target) {
    return ipcRenderer.sendSync('standalone-fs-delete', target);
  },
  ensureDir(target) {
    return ipcRenderer.sendSync('standalone-fs-ensure', target);
  }
});
