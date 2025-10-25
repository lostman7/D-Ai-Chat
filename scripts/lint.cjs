#!/usr/bin/env node

const { spawnSync } = require('node:child_process');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');

const projectRoot = process.cwd();

function runNodeCheck(file) {
  const result = spawnSync(process.execPath, ['--check', file], {
    cwd: projectRoot,
    stdio: 'inherit'
  });

  if (result.status !== 0) {
    throw new Error(`Syntax check failed for ${file}`);
  }
}

function verifyMarkers(file, markers) {
  const content = readFileSync(join(projectRoot, file), 'utf8');
  for (const marker of markers) {
    if (!content.includes(marker.pattern)) {
      throw new Error(`Expected to find ${marker.description} in ${file}`);
    }
  }
}

try {
  runNodeCheck('app.js');

  verifyMarkers('index.html', [
    { pattern: 'id=\"optionsHandle\"', description: 'options drawer handle' },
    { pattern: 'id=\"optionsDrawer\"', description: 'options drawer panel' },
    { pattern: 'id=\"arena_mode\"', description: 'arena selector' },
    { pattern: 'id=\"agent_list\"', description: 'agent roster' },
    { pattern: 'id=\"boardViewport\"', description: 'board viewport container' }
  ]);

  verifyMarkers('app.js', [
    { pattern: 'async function startArena()', description: 'arena bootstrapper' },
    { pattern: 'async function gatherBoardContext', description: 'board-aware retrieval' },
    { pattern: 'function buildPromptFromBoard', description: 'prompt synthesizer' },
    { pattern: 'async function callEmbedding', description: 'embedding request handler' }
  ]);

  verifyMarkers('styles.css', [
    { pattern: '.options-handle', description: 'options handle styles' },
    { pattern: '.side-panel', description: 'drawer styling' },
    { pattern: '.board-panel', description: 'board layout styles' }
  ]);

  console.log('All static checks passed.');
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
