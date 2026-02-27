#!/usr/bin/env sh
# init_vault.sh - Initialize and configure Vault for trading bot
# Run once after starting Vault for the first time
set -e

VAULT_ADDR="${VAULT_ADDR:-http://localhost:8200}"

echo "Waiting for Vault to start..."
until curl -s "$VAULT_ADDR/v1/sys/health" | grep -q '"initialized"'; do
  sleep 2
done

echo "Initializing Vault..."
INIT_OUTPUT=$(vault operator init -key-shares=5 -key-threshold=3 -format=json)

# Save keys securely (in production, distribute to key holders)
echo "$INIT_OUTPUT" > /tmp/vault-init.json
echo "⚠ SAVE THESE KEYS SECURELY:"
echo "$INIT_OUTPUT" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for i, key in enumerate(d['unseal_keys_b64']):
    print(f'  Unseal Key {i+1}: {key}')
print(f'  Root Token: {d[\"root_token\"]}')
"

# Auto-unseal for dev (do NOT do this in production)
ROOT_TOKEN=$(echo "$INIT_OUTPUT" | python3 -c "import json,sys; print(json.load(sys.stdin)['root_token'])")
UNSEAL_KEY_1=$(echo "$INIT_OUTPUT" | python3 -c "import json,sys; print(json.load(sys.stdin)['unseal_keys_b64'][0])")
UNSEAL_KEY_2=$(echo "$INIT_OUTPUT" | python3 -c "import json,sys; print(json.load(sys.stdin)['unseal_keys_b64'][1])")
UNSEAL_KEY_3=$(echo "$INIT_OUTPUT" | python3 -c "import json,sys; print(json.load(sys.stdin)['unseal_keys_b64'][2])")

vault operator unseal "$UNSEAL_KEY_1"
vault operator unseal "$UNSEAL_KEY_2"
vault operator unseal "$UNSEAL_KEY_3"

export VAULT_TOKEN="$ROOT_TOKEN"

# Enable KV v2 secrets engine for trading bot secrets
vault secrets enable -path=trading -version=2 kv
echo "✓ KV v2 secrets engine enabled at 'trading/'"

# Create policy for trading bot
vault policy write trading-bot - <<EOF
# Trading bot secrets policy
path "trading/data/*" {
  capabilities = ["read"]
}
path "trading/metadata/*" {
  capabilities = ["read", "list"]
}
EOF
echo "✓ Policy 'trading-bot' created"

# Create AppRole for trading bot authentication
vault auth enable approle
vault write auth/approle/role/trading-bot \
  token_policies="trading-bot" \
  token_ttl=1h \
  token_max_ttl=4h \
  secret_id_ttl=0

ROLE_ID=$(vault read -field=role_id auth/approle/role/trading-bot/role-id)
SECRET_ID=$(vault write -field=secret_id -f auth/approle/role/trading-bot/secret-id)

echo ""
echo "✓ AppRole configured"
echo "  VAULT_ROLE_ID=$ROLE_ID"
echo "  VAULT_SECRET_ID=$SECRET_ID"
echo ""
echo "Add these to your .env file to use Vault authentication."

# Seed placeholder secrets (update with real values)
vault kv put trading/bot/exchange \
  binance_api_key="REPLACE_WITH_REAL_KEY" \
  binance_api_secret="REPLACE_WITH_REAL_SECRET"

vault kv put trading/bot/database \
  password="REPLACE_WITH_DB_PASSWORD"

vault kv put trading/bot/telegram \
  token="REPLACE_WITH_BOT_TOKEN"

echo "✓ Placeholder secrets seeded at 'trading/bot/'"
echo ""
echo "Vault initialization complete!"
