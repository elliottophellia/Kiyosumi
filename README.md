# Kiyosumi

Kiyosumi is a native Oh My Pi extension for durable memory and retrieval. It stores memory and indexed passages in an embedded PGlite database, then adds relevant context to the next model turn.

The extension uses Oh My Pi's public `ExtensionAPI`. It does not patch the agent or change session files. Voyage AI is the default embedding and reranking provider, while custom embedding and reranking endpoints are supported independently.

## Requirements

- Oh My Pi 18.3.1 or newer
- Bun 1.2 or newer
- A Voyage API key for the default provider, or credentials for custom endpoints

Memory works without a provider. Retrieval tools report a missing provider configuration until an embedding endpoint is available.

## Install

Install the published package through Oh My Pi:

```sh
omp plugin install @elliottophellia/kiyosumi
```

Restart Oh My Pi after installation. The package is also available on [npm](https://www.npmjs.com/package/@elliottophellia/kiyosumi).

For local development, link a checkout instead:

```sh
bun install
omp plugin link . --scope project
```

To load a checkout for one session:

```sh
omp --plugin-dir /path/to/Kiyosumi
```

## Providers

The default configuration uses Voyage-compatible endpoints:

```text
Embedding: https://api.voyageai.com/v1/embeddings
Rerank:    https://api.voyageai.com/v1/rerank
```

`VOYAGE_API_KEY` is the default credential for both requests. Override the endpoints, models, styles, and keys independently when a provider needs different settings.

### Custom embedding endpoint

```sh
export KIYOSUMI_PROVIDER_NAME=acme
export KIYOSUMI_EMBEDDING_ENDPOINT=https://embed.example/v1/embeddings
export KIYOSUMI_EMBEDDING_API_KEY=your-embedding-key
export KIYOSUMI_EMBEDDING_MODEL=acme-embed
export KIYOSUMI_EMBEDDING_API_STYLE=openai
```

`KIYOSUMI_EMBEDDING_API_STYLE` accepts `voyage` or `openai`. The OpenAI-compatible style sends `input`, `model`, and `encoding_format`. The Voyage style adds `input_type`, `output_dimension`, and `truncation`.

### Custom rerank endpoint

```sh
export KIYOSUMI_RERANK_ENDPOINT=https://rerank.example/v1/rerank
export KIYOSUMI_RERANK_API_KEY=your-rerank-key
export KIYOSUMI_RERANK_MODEL=acme-rerank
export KIYOSUMI_RERANK_API_STYLE=generic
```

`KIYOSUMI_RERANK_API_STYLE` accepts `voyage` or `generic`. The response parser accepts `data` or `results`, with `relevance_score` or `score` on each item.

`KIYOSUMI_EMBEDDING_BASE_URL` and `KIYOSUMI_RERANK_BASE_URL` remain supported as base-URL aliases. Kiyosumi appends the matching operation path when a base URL is supplied. Prefer the `*_ENDPOINT` variables when a custom service uses a different path.

## Configuration

Kiyosumi reads configuration from the environment when the extension loads.

| Variable | Default | Purpose |
| --- | --- | --- |
| `VOYAGE_API_KEY` | unset | Default credential for Voyage endpoints |
| `KIYOSUMI_DATA_DIR` | profile data directory | PGlite data directory |
| `KIYOSUMI_PROVIDER_NAME` | `voyage` | Provider identity used in collection metadata |
| `KIYOSUMI_EMBEDDING_ENDPOINT` | Voyage embeddings URL | Full embedding request URL |
| `KIYOSUMI_EMBEDDING_API_KEY` | `VOYAGE_API_KEY` | Embedding credential |
| `KIYOSUMI_EMBEDDING_MODEL` | `voyage-4` | Embedding model |
| `KIYOSUMI_EMBEDDING_API_STYLE` | `voyage` | `voyage` or `openai` request format |
| `KIYOSUMI_EMBED_DIMENSIONS` | `1024` | One of `256`, `512`, `1024`, or `2048` |
| `KIYOSUMI_RERANK_ENDPOINT` | Voyage rerank URL | Full rerank request URL |
| `KIYOSUMI_RERANK_API_KEY` | `VOYAGE_API_KEY` | Rerank credential |
| `KIYOSUMI_RERANK_MODEL` | `rerank-2.5` | Rerank model |
| `KIYOSUMI_RERANK_API_STYLE` | `voyage` | `voyage` or `generic` request format |
| `KIYOSUMI_RERANK` | `1` | Enable reranking |
| `KIYOSUMI_HYBRID` | `1` | Combine dense and PostgreSQL lexical recall |
| `KIYOSUMI_DEDUPE` | `1` | Remove identical trimmed passages |
| `KIYOSUMI_CHUNK_SIZE` | `1200` | Characters per indexed chunk |
| `KIYOSUMI_CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `KIYOSUMI_RECALL` | `40` | Candidates recalled before reranking |
| `KIYOSUMI_TOP_K` | `8` | Maximum passages returned to the model |
| `KIYOSUMI_MEMORY_ENABLED` | `1` | Enable durable memory tools and prompt injection |
| `KIYOSUMI_MEMORY_CHAR_LIMIT` | `12000` | Maximum memory block size in characters |
| `KIYOSUMI_AUTO_CONTEXT` | `1` | Index completed conversation exchanges and inject relevant saved/project/conversation context |
| `KIYOSUMI_CONTEXT_MAX_CHARS` | `4000` | Retrieved context budget per turn |
| `KIYOSUMI_CONTEXT_MAX_PASSAGES` | `6` | Retrieved passages per turn |
| `KIYOSUMI_MAX_FILE_BYTES` | `2097152` | Largest file considered for indexing |
| `KIYOSUMI_MAX_FILES` | `3000` | Largest workspace file count per index operation |
| `KIYOSUMI_MAX_INDEX_BYTES` | `67108864` | Largest total text size per index operation |

Project files are **not** scanned automatically at session start. Run `/kiyosumi analyze` to request a bounded workspace index and a read-only project analysis. Analysis is still sent if indexing fails or no embedding provider is configured. The resulting project overview should be saved through `kiyosumi_memory` as project memory with key `project-overview`.

`KIYOSUMI_AUTO_INDEX` is no longer supported. Use `/kiyosumi index` or `/kiyosumi analyze` to explicitly index project source.

Boolean variables accept `1`, `0`, `true`, `false`, `yes`, `no`, `on`, and `off`.

Without `KIYOSUMI_DATA_DIR`, the PGlite data directory follows the active Oh My Pi profile. `OMP_PROFILE` selects a profile-specific directory. `PI_CODING_AGENT_DIR` takes precedence when it is set.

## Use

### Tools

- `kiyosumi_memory` saves, searches, lists, or deletes durable facts.
- `kiyosumi_rag_search` searches the current project collection by default; advanced calls may select a collection.
- `kiyosumi_rag_index` indexes a file or directory inside the current workspace; advanced calls may select a collection and include glob.

### Command

All human slash actions use `/kiyosumi`; the project collection is inferred from the current workspace.

```text
/kiyosumi help
/kiyosumi analyze
/kiyosumi index [path]
/kiyosumi search <query>
/kiyosumi memory [list]
/kiyosumi memory search <query>
/kiyosumi memory save <text>
/kiyosumi memory delete <key-or-id>
/kiyosumi status
/kiyosumi delete-index [collection]
```

`index` defaults to the whole workspace. Re-indexing the whole workspace replaces only its project collection. Indexing a subpath updates those documents and leaves unrelated indexed files intact. `search` reports how to index when the current project collection has no matching passages. `memory save` accepts `key: value` to retain the stable-key upsert behavior. `delete-index` defaults to the current project collection; an optional named collection is allowed only for this destructive action.

`analyze` attempts a bounded full-workspace index, then asks the agent to inspect key project files without making edits and summarize purpose, stack, organization, build/run/test, and notable details. If the provider is missing or indexing fails, the analysis prompt is still delivered. The overview is requested as project memory `project-overview`.

There are no separate `/kiyosumi-memory`, `/kiyosumi-remember`, `/kiyosumi-forget`, or `-rag-*` slash commands. The namespaced agent tools remain available for advanced use.

## Retrieval pipeline

Kiyosumi stores memories and chunks in PGlite, an embedded PostgreSQL database. Each collection records its vector dimensions, provider identity, and revision counter. A model, endpoint, or dimension change is rejected for an existing collection. Delete the collection and index it again before changing its vector space.

A search uses these stages:

1. The configured embedding provider embeds the query.
2. Document chunks use the configured document embedding path.
3. Exact cosine recall runs over the collection's cached vectors.
4. PostgreSQL `tsvector` and GIN indexes provide lexical recall, with an `ILIKE` fallback.
5. Reciprocal-rank fusion combines dense and lexical candidates.
6. The configured rerank endpoint reorders candidates when enabled.
7. Exact trimmed-content deduplication runs before the `top_k` limit.

Completed conversation exchanges are stored in `kiyosumi-conversations` when `KIYOSUMI_AUTO_CONTEXT=1`. Project source is only indexed through explicit `/kiyosumi index` or `/kiyosumi analyze` actions; no session-start project scan occurs.

## Storage and safety

- PGlite owns the local data directory. The plugin does not require a separate database server.
- Public store operations are serialized because PGlite is a single-connection embedded database.
- The API keys are read from the environment and are never written to the database or status output.
- Status output redacts endpoint query strings and URL credentials.
- Indexing skips symlinks, binary or invalid UTF-8 files, common dependency directories, and files over the configured size limit.
- Retrieved passages and memory entries are labeled as reference material in the model prompt.
- `kiyosumi_rag_index` cannot follow a target outside the current workspace.
- A failed embedding request does not replace the previous stored version of a document.
- Kiyosumi owns its PGlite memory store. It does not write through Oh My Pi's optional memory backend.

## Development

```sh
bun install
bun run typecheck
bun test
bun run build
```

Run the complete verification sequence with:

```sh
bun run verify
```

The tests cover configuration, custom provider request formats, PGlite persistence, memory updates, PostgreSQL lexical search, hybrid retrieval, reranking, deduplication, vector identity checks, stale-chunk pruning, reopened databases, and async extension lifecycle behavior.
