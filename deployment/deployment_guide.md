# Deployment Guide (Overview)

1) Secrets: store API keys and credentials in AWS Secrets Manager or HashiCorp Vault. Inject into containers as environment variables or fetch at runtime.

2) Docker Compose for local: `docker-compose.yml` included. For production, use Kubernetes with manifests in `k8s/` (create per-environment).

3) Nodes: run 3+ replicas across AWS, Hetzner and a local host. Use a load balancer and health checks.

4) Failover: HealthMonitor writes heartbeat metrics; orchestrator should promote backup if latency/health degrade.

5) Observability: expose Prometheus metrics on port 8001, Grafana dashboard to visualize.

6) Paper trading: enable paper mode in config to avoid live orders until validated.
