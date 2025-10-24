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
