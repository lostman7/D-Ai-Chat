#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const colors = {
  reset: '\u001b[0m',
  blue: '\u001b[34m',
  green: '\u001b[32m',
  yellow: '\u001b[33m',
  magenta: '\u001b[35m',
  cyan: '\u001b[36m',
  orange: '\u001b[38;5;208m',
  pink: '\u001b[38;5;213m',
  gray: '\u001b[38;5;244m'
};

const roleColors = new Map([
  ['SAM-A', colors.blue],
  ['SAM-B', colors.green],
  ['USER', colors.yellow],
  ['SYSTEM', colors.magenta],
  ['Reasoning', colors.cyan]
]);

const stopTerms = new Set(['Reasoning', 'Turn', 'Conversation', 'Exported', 'Insights', 'Questions', 'Next', 'moves']);

function colorizeLine(line, knownTerms) {
  let output = line;
  for (const [role, color] of roleColors.entries()) {
    if (output.includes(role)) {
      output = color + output + colors.reset;
      break;
    }
  }

  output = output.replace(/Reasoning:/gi, (match) => `${colors.cyan}${match}${colors.reset}`);
  output = output.replace(/([A-Za-z0-9_]+\s*[+\-/*=]\s*[A-Za-z0-9_]+)/g, (match) => `${colors.orange}${match}${colors.reset}`);

  output = output.replace(/\b([A-Z][A-Za-z0-9_-]{3,})\b/g, (match) => {
    if (stopTerms.has(match) || knownTerms.has(match)) {
      return match;
    }
    knownTerms.add(match);
    return `${colors.pink}${match}${colors.reset}`;
  });

  return output;
}

function processFile(filePath) {
  const absolute = path.resolve(process.cwd(), filePath);
  if (!fs.existsSync(absolute)) {
    console.error(`${colors.orange}File not found:${colors.reset} ${filePath}`);
    process.exitCode = 1;
    return;
  }
  const content = fs.readFileSync(absolute, 'utf8');
  const lines = content.split(/\r?\n/);
  const knownTerms = new Set(stopTerms);
  console.log(`${colors.gray}─ Replay phaser :: ${filePath}${colors.reset}`);
  for (const line of lines) {
    if (!line.trim()) {
      console.log('');
      continue;
    }
    console.log(colorizeLine(line, knownTerms));
  }
}

function main() {
  const [, , ...args] = process.argv;
  if (args.length === 0) {
    console.error('Usage: node scripts/replay-phaser.js <transcript.txt> [more.txt]');
    process.exit(1);
    return;
  }
  for (const file of args) {
    processFile(file);
  }
}

main();
