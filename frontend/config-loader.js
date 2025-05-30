const fs = require('fs');
const path = require('path');

const CONFIG_DIR = path.join(__dirname, '..', 'config');
const ENV = process.env.ENV || 'dev';
const CONFIG_PATH = path.join(CONFIG_DIR, `${ENV}.yaml`);

const yaml = require('js-yaml');

function loadConfig() {
  return yaml.load(fs.readFileSync(CONFIG_PATH, 'utf8'));
}

module.exports = { loadConfig };
