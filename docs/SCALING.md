# Scaling And Load Balancing

This guide explains how to enable and operate the optional horizontal scaling layer that lives in
`src/core/scaling.py`. The layer keeps track of active bot instances, picks the least loaded node for
each request, and (optionally) pins user sessions to the same node via Redis.

## Components
- `ServiceRegistry` (`src/core/scaling.py:33`) stores metadata for every running node in Redis and
  emits heartbeats so the cluster can spot stalled instances. When Redis is unavailable it falls back
  to an in-process dictionary to keep the bot running in single-node mode.
- `LoadBalancer` (`src/core/scaling.py:174`) reads the registry and selects a destination node using
  round-robin, least-load, or random strategies. It also exposes statistics for observability.
- `SessionAffinity` (`src/core/scaling.py:436`) keeps a `session_id -> node_id` mapping so a user
  stays on the same worker while their subscription or payment flow is running. Redis is preferred,
  but the class keeps a process-local fallback in case Redis is down.
- `ScalingManager` (`src/core/scaling.py:489`) orchestrates routing and exposes `should_scale_out`
  / `should_scale_in` helpers that operators can wire into their auto-scaling logic.

`src/core/bot_app/startup.py:205` wires those components automatically when
`AppSettings.enable_scaling` is true (see below).

## Configuration
Set these environment variables in `.env` (or secrets manager) before starting the bot:

| Variable | Purpose |
|----------|---------|
| `ENABLE_SCALING=1` | Turns on the scaling block inside `startup.py`. |
| `REDIS_URL=redis://host:port/0` | Required for the service registry and session pinning. |
| `HEARTBEAT_INTERVAL` (seconds) | Overrides how often each node publishes its status (default 15). |
| `SESSION_AFFINITY_TTL` (seconds) | Time-to-live for the session to node mapping (default 3600). |

If Redis is missing the bot keeps running but the registry and affinity fall back to local storage,
so other nodes will not discover the instance. Always verify Redis connectivity before setting
`ENABLE_SCALING`.

## Runtime lifecycle
1. `startup.py` builds the registry, starts the heartbeat and cleanup tasks, then instantiates the
   load balancer and session affinity helpers.
2. `ScalingManager.route_request(session_id=...)` can be called from handlers or background jobs to
   pick the target node for a heavy task. When a `session_id` is provided the manager checks the
   affinity table first.
3. `ScalingManager.get_cluster_status()` returns a diagnostic payload that can be exposed through
   admin commands or a status endpoint. It tells you how many nodes are healthy, their average load,
   and whether auto-scaling thresholds have been crossed.
4. When the process shuts down, `ServiceRegistry.stop_background_tasks()` is called from
   `startup.py` so the node removes itself from Redis.

## Operations and troubleshooting
- Use `redis-cli hgetall aivan:nodes` (the default key prefix) to inspect the raw registry when
  debugging connectivity.
- If you need deterministic routing (for example for court case uploads), pass an explicit
  `session_id` to `route_request`. Without it, the load balancer may move the user between nodes.
- Tune `ScalingManager.max_load_threshold`, `min_nodes`, and `max_nodes` when wiring the manager into
  an auto-scaling controller. The defaults (80% load, 1-10 nodes) are intentionally conservative.
- When Redis goes down the registry logs warnings and uses the in-memory map. Bring Redis back online
  and restart the bot to restore cluster visibility.

The scaling module is purposely decoupled from aiogram so you can reuse it in other background
services (for example, OCR workers). Import the same classes and provide your own heartbeat or
routing hooks if needed.
