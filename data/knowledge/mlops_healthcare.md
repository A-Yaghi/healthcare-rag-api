# MLOps Notes for Healthcare AI

Shipping an AI model in healthcare requires more than a notebook. Useful production
practices include versioned datasets, experiment tracking, containerized inference
APIs, automated tests, and rollback plans.

FastAPI is a common choice for Python model services because it provides typed
request schemas, OpenAPI docs, and low-overhead async endpoints. API keys or OAuth
protect internal tools. Metrics such as average and p95 latency help teams detect
regressions after model updates.

For Belgian hospital and health-tech environments, documentation quality, GDPR
awareness, and clear audit trails matter as much as raw accuracy. Start with a
narrow use case, measure real workflow impact, and expand only after monitoring is
in place.
