# HashiCorp Vault Configuration for Trading Bot
# Development / local configuration
# For production, use Vault Helm chart or official Vault operator

ui = true
cluster_addr  = "http://0.0.0.0:8201"
api_addr      = "http://0.0.0.0:8200"
disable_mlock = true

# Storage backend (file-based for dev; use Consul/Raft for production)
storage "file" {
  path = "/vault/data"
}

# TCP listener
listener "tcp" {
  address       = "0.0.0.0:8200"
  tls_disable   = true   # IMPORTANT: enable TLS in production!
}

# Telemetry
telemetry {
  prometheus_retention_time = "30s"
  disable_hostname = true
}
