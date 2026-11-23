# Design decisions

Format: **Decision** — why — alternative rejected.

- **Official APIs only, no UI scraping.** Stays within provider ToS and can be reproduced; API answers with web search on approximate consumer answers. Rejected: headless-browser scraping of chatgpt.com (fragile, against ToS). Limitation noted in README.
- **SQLite for tests, Postgres for dev/prod.** Tests run on Windows with no Docker. Rejected: testcontainers (slow and fiddly on Windows).
- **Embeddings stored as arrays in a regular column, similarity done in numpy.** Panel sizes are thousands, not millions. Rejected: pgvector (extra setup); revisit if scale grows.
- **Mock mode is the default.** CI and the public demo never spend money; fixtures make results deterministic.
