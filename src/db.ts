import { createHash } from "node:crypto";
import { mkdir } from "node:fs/promises";
import { PGlite, type PGliteInterface, type Transaction } from "@electric-sql/pglite";
import type {
	MemoryRecord,
	MemoryRepository,
	MemoryScope,
	RagChunk,
	RagHit,
	SaveMemoryInput,
} from "./types.js";

type QueryExecutor = Pick<PGliteInterface, "query"> | Pick<Transaction, "query">;

type NumericValue = number | bigint | string;

interface MemoryRow {
	id: string;
	scope: MemoryScope;
	scope_key: string;
	mem_key: string;
	content: string;
	source: string;
	pinned: boolean | number;
	created_at: NumericValue;
	updated_at: NumericValue;
}

interface ChunkRow {
	id: string;
	collection: string;
	doc_id: string;
	path: string;
	chunk_index: number;
	content: string;
	embedding: unknown;
	dimensions: NumericValue;
	meta: unknown;
}

interface CollectionStateRow {
	dimensions: NumericValue;
	model: string;
	revision: NumericValue;
}

interface WorkspaceRow {
	fingerprint: string;
	file_count: NumericValue;
	chunk_count: NumericValue;
}

export interface CollectionState {
	dimensions: number;
	revision: number;
	model: string;
}

export class EmbeddingModelMismatchError extends Error {
	constructor(
		readonly collection: string,
		readonly expected: string,
		readonly received: string,
	) {
		super(`embedding model mismatch for collection ${collection}: expected ${expected || "an unknown model"}, received ${received}`);
		this.name = "EmbeddingModelMismatchError";
	}
}

interface CachedChunk {
	id: string;
	documentId: string;
	path: string;
	content: string;
	meta: Record<string, unknown>;
	vector: Float32Array;
}

interface VectorCache {
	revision: number;
	dimensions: number;
	model: string;
	chunks: CachedChunk[];
}

export interface ReplaceOptions {
	documentIds?: readonly string[];
	replaceCollection?: boolean;
}

export class DimensionMismatchError extends Error {
	constructor(
		readonly expected: number,
		readonly received: number,
	) {
		super(`vector dimension mismatch: collection expects ${expected}, received ${received}`);
		this.name = "DimensionMismatchError";
	}
}

export class EmbeddingShapeError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "EmbeddingShapeError";
	}
}

function hashId(value: string, bytes = 12): string {
	return createHash("sha256").update(value).digest("hex").slice(0, bytes * 2);
}

function normalizeMemoryKey(key: string, content: string): string {
	const trimmed = key.trim();
	if (trimmed) return trimmed.slice(0, 160);
	return `memory-${hashId(content, 8)}`;
}

function parseJsonValue(value: unknown): unknown {
	if (typeof value !== "string") return value;
	try {
		return JSON.parse(value) as unknown;
	} catch {
		return value;
	}
}

function parseJsonObject(value: unknown): Record<string, unknown> {
	const parsed = parseJsonValue(value);
	return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)
		? parsed as Record<string, unknown>
		: {};
}

function numberValue(value: unknown, label: string): number {
	const result = typeof value === "number" ? value : typeof value === "bigint" ? Number(value) : Number(value);
	if (!Number.isFinite(result)) throw new EmbeddingShapeError(`${label} is not a finite number`);
	return result;
}

function vectorFromJson(value: unknown, dimensions: number): Float32Array {
	const parsed = parseJsonValue(value);
	if (!Array.isArray(parsed) || parsed.length !== dimensions) {
		throw new EmbeddingShapeError(`stored vector has ${Array.isArray(parsed) ? parsed.length : 0} values, expected ${dimensions}`);
	}
	const result = new Float32Array(dimensions);
	for (let index = 0; index < dimensions; index++) {
		const item = parsed[index];
		if (typeof item !== "number" && typeof item !== "string") {
			throw new EmbeddingShapeError("stored vector contains a non-numeric value");
		}
		const value = Number(item);
		if (!Number.isFinite(value)) throw new EmbeddingShapeError("stored vector contains a non-finite value");
		result[index] = value;
	}
	return result;
}

function finiteNonZeroVector(vector: Float32Array): void {
	if (vector.length === 0) throw new EmbeddingShapeError("embedding vector is empty");
	let nonZero = false;
	for (const value of vector) {
		if (!Number.isFinite(value)) throw new EmbeddingShapeError("embedding vector contains a non-finite value");
		if (value !== 0) nonZero = true;
	}
	if (!nonZero) throw new EmbeddingShapeError("embedding vector is all zero");
}

function normalizeVector(vector: Float32Array): Float32Array {
	let norm = 0;
	for (const value of vector) norm += value * value;
	if (!Number.isFinite(norm) || norm === 0) throw new EmbeddingShapeError("embedding vector cannot be normalized");
	const result = new Float32Array(vector.length);
	const inverse = 1 / Math.sqrt(norm);
	for (let index = 0; index < vector.length; index++) result[index] = vector[index]! * inverse;
	return result;
}

function cosine(left: Float32Array, right: Float32Array): number {
	if (left.length !== right.length) throw new DimensionMismatchError(left.length, right.length);
	let score = 0;
	for (let index = 0; index < left.length; index++) score += left[index]! * right[index]!;
	return Number.isFinite(score) ? score : 0;
}

function escapeLike(value: string): string {
	return value.replaceAll("\\", "\\\\").replaceAll("%", "\\%").replaceAll("_", "\\_");
}

function postgresFtsQuery(query: string): string | undefined {
	const terms = query.normalize("NFKC").match(/[\p{L}\p{N}_-]+/gu) ?? [];
	if (terms.length === 0) return undefined;
	return terms.slice(0, 32).map((term) => `'${term}':*`).join(" | ");
}

function boundedLimit(value: number, minimum: number, maximum: number): number {
	const finite = Number.isFinite(value) ? Math.trunc(value) : minimum;
	return Math.min(Math.max(finite, minimum), maximum);
}

function changedRows(result: { affectedRows?: number; rowCount?: number }): number {
	return result.affectedRows ?? result.rowCount ?? 0;
}

function rowToMemory(row: MemoryRow): MemoryRecord {
	return {
		id: row.id,
		scope: row.scope,
		scopeKey: row.scope_key,
		key: row.mem_key,
		content: row.content,
		source: row.source,
		pinned: row.pinned === true || row.pinned === 1,
		createdAt: numberValue(row.created_at, "memory creation timestamp"),
		updatedAt: numberValue(row.updated_at, "memory update timestamp"),
	};
}

function rowToChunk(row: ChunkRow): CachedChunk {
	const dimensions = numberValue(row.dimensions, "stored vector dimensions");
	return {
		id: row.id,
		documentId: row.doc_id,
		path: row.path,
		content: row.content,
		meta: parseJsonObject(row.meta),
		vector: vectorFromJson(row.embedding, dimensions),
	};
}

function hitKey(hit: Pick<RagHit, "documentId" | "path" | "content">): string {
	return `${hit.documentId}\u0000${hit.path}\u0000${hit.content}`;
}

export class KiyosumiStore implements MemoryRepository {
	private readonly vectorCache = new Map<string, VectorCache>();
	private operationTail: Promise<void> = Promise.resolve();
	private closeRequested = false;
	private closed = false;
	private closePromise: Promise<void> | undefined;
	private memoryFtsAvailable = false;
	private ragFtsAvailable = false;

	private constructor(private readonly db: PGliteInterface) {}

	static async open(dataDir: string): Promise<KiyosumiStore> {
		await mkdir(dataDir, { recursive: true, mode: 0o700 });
		const db = await PGlite.create(dataDir);
		try {
			await db.waitReady;
			const store = new KiyosumiStore(db);
			await store.createSchema();
			return store;
		} catch (error) {
			await db.close();
			throw error;
		}
	}

	private enqueue<T>(operation: () => Promise<T>): Promise<T> {
		if (this.closeRequested || this.closed) return Promise.reject(new Error("KiyosumiStore is closed"));
		const result = this.operationTail.then(operation);
		this.operationTail = result.then(() => undefined, () => undefined);
		return result;
	}

	async close(): Promise<void> {
		if (this.closePromise) return this.closePromise;
		this.closeRequested = true;
		const result = this.operationTail.then(async () => {
			this.vectorCache.clear();
			await this.db.close();
			this.closed = true;
		});
		this.operationTail = result.then(() => undefined, () => undefined);
		this.closePromise = result;
		return result;
	}

	private async createSchema(): Promise<void> {
		await this.db.exec(`
			CREATE TABLE IF NOT EXISTS memories (
				id TEXT PRIMARY KEY,
				scope TEXT NOT NULL,
				scope_key TEXT NOT NULL DEFAULT '',
				mem_key TEXT NOT NULL,
				content TEXT NOT NULL,
				source TEXT NOT NULL DEFAULT 'agent',
				pinned BOOLEAN NOT NULL DEFAULT FALSE,
				created_at BIGINT NOT NULL,
				updated_at BIGINT NOT NULL,
				UNIQUE(scope, scope_key, mem_key)
			);
			CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope, scope_key, updated_at DESC, pinned DESC);
			CREATE TABLE IF NOT EXISTS rag_collections (
				collection TEXT PRIMARY KEY,
				dimensions INTEGER NOT NULL DEFAULT 0,
				model TEXT NOT NULL DEFAULT '',
				revision BIGINT NOT NULL DEFAULT 0,
				created_at BIGINT NOT NULL,
				updated_at BIGINT NOT NULL
			);
			CREATE TABLE IF NOT EXISTS rag_chunks (
				id TEXT PRIMARY KEY,
				collection TEXT NOT NULL,
				doc_id TEXT NOT NULL,
				path TEXT NOT NULL DEFAULT '',
				chunk_index INTEGER NOT NULL DEFAULT 0,
				content TEXT NOT NULL,
				embedding JSONB NOT NULL,
				dimensions INTEGER NOT NULL,
				meta JSONB NOT NULL DEFAULT '{}'::jsonb,
				created_at BIGINT NOT NULL,
				updated_at BIGINT NOT NULL,
				UNIQUE(collection, doc_id, chunk_index)
			);
			CREATE INDEX IF NOT EXISTS idx_rag_chunks_collection ON rag_chunks(collection);
			CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc ON rag_chunks(collection, doc_id, chunk_index);
			CREATE TABLE IF NOT EXISTS workspace_state (
				path TEXT PRIMARY KEY,
				fingerprint TEXT NOT NULL,
				file_count BIGINT NOT NULL DEFAULT 0,
				chunk_count BIGINT NOT NULL DEFAULT 0,
				indexed_at BIGINT NOT NULL
			);
		`);
		try {
			await this.db.exec("CREATE INDEX IF NOT EXISTS idx_memories_fts ON memories USING GIN (to_tsvector('simple', content));");
			this.memoryFtsAvailable = true;
		} catch {
			this.memoryFtsAvailable = false;
		}
		try {
			await this.db.exec("CREATE INDEX IF NOT EXISTS idx_rag_chunks_fts ON rag_chunks USING GIN (to_tsvector('simple', content));");
			this.ragFtsAvailable = true;
		} catch {
			this.ragFtsAvailable = false;
		}
	}

	async save(input: SaveMemoryInput): Promise<MemoryRecord> {
		return this.enqueue(async () => {
			const scope: MemoryScope = input.scope;
			const scopeKey = input.scopeKey.trim();
			const content = input.content.trim();
			if (!content) throw new Error("memory content cannot be empty");
			if (scope !== "global" && scope !== "project" && scope !== "session") {
				throw new Error(`unsupported memory scope: ${input.scope}`);
			}
			const key = normalizeMemoryKey(input.key, content);
			const id = `mem_${hashId(`${scope}|${scopeKey}|${key}`, 10)}`;
			const now = Date.now();
			const source = input.source?.trim() || "agent";
			return this.db.transaction(async (tx) => {
				await tx.query(`
					INSERT INTO memories(id, scope, scope_key, mem_key, content, source, pinned, created_at, updated_at)
					VALUES ($1, $2, $3, $4, $5, $6, FALSE, $7, $8)
					ON CONFLICT(id) DO UPDATE SET
						content = EXCLUDED.content,
						source = EXCLUDED.source,
						updated_at = EXCLUDED.updated_at
				`, [id, scope, scopeKey, key, content, source, now, now]);
				const result = await tx.query<MemoryRow>("SELECT * FROM memories WHERE id = $1", [id]);
				const row = result.rows[0];
				if (!row) throw new Error("memory disappeared after save");
				return rowToMemory(row);
			});
		});
	}

	async get(id: string): Promise<MemoryRecord | undefined> {
		return this.enqueue(async () => {
			const result = await this.db.query<MemoryRow>("SELECT * FROM memories WHERE id = $1", [id]);
			const row = result.rows[0];
			return row ? rowToMemory(row) : undefined;
		});
	}

	async list(scope?: MemoryScope, scopeKey?: string, limit = 200): Promise<MemoryRecord[]> {
		return this.enqueue(() => this.listInternal(scope, scopeKey, limit));
	}

	async search(query: string, limit = 30): Promise<MemoryRecord[]> {
		return this.enqueue(async () => {
			const text = query.trim();
			const bounded = boundedLimit(limit, 1, 200);
			if (!text) return this.listInternal(undefined, undefined, bounded);
			const ftsQuery = postgresFtsQuery(text);
			if (this.memoryFtsAvailable && ftsQuery) {
				try {
					const result = await this.db.query<MemoryRow>(`
						SELECT m.*, ts_rank_cd(to_tsvector('simple', m.content), to_tsquery('simple', $1)) AS relevance
						FROM memories m
						WHERE to_tsvector('simple', m.content) @@ to_tsquery('simple', $1)
						ORDER BY relevance DESC, m.pinned DESC, m.updated_at DESC, m.id
						LIMIT $2
					`, [ftsQuery, bounded]);
					return result.rows.map(rowToMemory);
				} catch {
					// Fall through to a predictable ILIKE search for malformed or unavailable FTS.
				}
			}
			const result = await this.db.query<MemoryRow>(`
				SELECT * FROM memories
				WHERE content ILIKE $1 ESCAPE '\\' OR mem_key ILIKE $2 ESCAPE '\\'
				ORDER BY pinned DESC, updated_at DESC, id
				LIMIT $3
			`, [`%${escapeLike(text.toLowerCase())}%`, `%${escapeLike(text.toLowerCase())}%`, bounded]);
			return result.rows.map(rowToMemory);
		});
	}

	async delete(id: string): Promise<boolean> {
		return this.enqueue(async () => this.db.transaction(async (tx) => {
			const result = await tx.query("DELETE FROM memories WHERE id = $1", [id]);
			return changedRows(result) > 0;
		}));
	}

	async reserveCollection(collection: string, dimensions: number, model: string, now = Date.now()): Promise<void> {
		return this.enqueue(() => this.db.transaction(async (tx) => this.reserveCollectionTx(tx, collection, dimensions, model, now)));
	}

	async assertEmbeddingModel(collection: string, model: string): Promise<void> {
		return this.enqueue(async () => {
			const state = await this.readCollectionState(this.db, collection);
			if (!state || state.dimensions === 0) return;
			if (state.model !== model) throw new EmbeddingModelMismatchError(collection, state.model, model);
		});
	}

	async replaceDocuments(collection: string, chunks: readonly RagChunk[], embeddingModel: string, options: ReplaceOptions = {}): Promise<void> {
		return this.enqueue(async () => {
			if (!collection.trim()) throw new Error("collection cannot be empty");
			const documentIds = [...new Set((options.documentIds ?? chunks.map((chunk) => chunk.documentId)).map((id) => id.trim()).filter(Boolean))];
			const byDocument = new Map<string, RagChunk[]>();
			for (const chunk of chunks) {
				if (chunk.collection !== collection) throw new Error(`chunk ${chunk.id} belongs to another collection`);
				if (!chunk.documentId.trim()) throw new Error(`chunk ${chunk.id} has no document id`);
				finiteNonZeroVector(chunk.embedding);
				const list = byDocument.get(chunk.documentId);
				if (list) list.push(chunk);
				else byDocument.set(chunk.documentId, [chunk]);
			}
			const dimensions = chunks[0]?.embedding.length;
			if (dimensions !== undefined) {
				for (const chunk of chunks) {
					if (chunk.embedding.length !== dimensions) throw new DimensionMismatchError(dimensions, chunk.embedding.length);
				}
			}
			await this.db.transaction(async (tx) => {
				if (dimensions !== undefined) await this.reserveCollectionTx(tx, collection, dimensions, embeddingModel);
				let changed = false;
				if (options.replaceCollection) {
					const deleted = await tx.query("DELETE FROM rag_chunks WHERE collection = $1", [collection]);
					if (changedRows(deleted) > 0) changed = true;
				}
				const seenIds = new Set<string>();
				const now = Date.now();
				for (const chunk of chunks) {
					if (seenIds.has(chunk.id)) throw new Error(`duplicate chunk id ${chunk.id} in batch`);
					seenIds.add(chunk.id);
					const existing = await tx.query<{ id: string; collection: string }>("SELECT id, collection FROM rag_chunks WHERE id = $1", [chunk.id]);
					const existingRow = existing.rows[0];
					if (existingRow && existingRow.collection !== collection) {
						throw new Error(`chunk ${existingRow.id} already belongs to collection ${existingRow.collection}`);
					}
					await tx.query(`
						INSERT INTO rag_chunks(id, collection, doc_id, path, chunk_index, content, embedding, dimensions, meta, created_at, updated_at)
						VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9::jsonb, $10, $11)
						ON CONFLICT(id) DO UPDATE SET
							doc_id = EXCLUDED.doc_id,
							path = EXCLUDED.path,
							chunk_index = EXCLUDED.chunk_index,
							content = EXCLUDED.content,
							embedding = EXCLUDED.embedding,
							dimensions = EXCLUDED.dimensions,
							meta = EXCLUDED.meta,
							updated_at = EXCLUDED.updated_at
						WHERE rag_chunks.collection = EXCLUDED.collection
					`, [
						chunk.id,
						collection,
						chunk.documentId,
						chunk.path,
						chunk.index,
						chunk.content,
						JSON.stringify(Array.from(chunk.embedding)),
						chunk.embedding.length,
						JSON.stringify(chunk.meta),
						now,
						now,
					]);
					changed = true;
				}
				for (const documentId of documentIds) {
					const maxIndex = (byDocument.get(documentId) ?? []).reduce((max, chunk) => Math.max(max, chunk.index), -1);
					const deleted = await tx.query("DELETE FROM rag_chunks WHERE collection = $1 AND doc_id = $2 AND chunk_index > $3", [collection, documentId, maxIndex]);
					if (changedRows(deleted) > 0) changed = true;
				}
				if (changed) await this.bumpCollectionRevision(tx, collection, now);
			});
			this.vectorCache.delete(collection);
		});
	}

	async deleteDocuments(collection: string, documentIds: readonly string[]): Promise<number> {
		return this.enqueue(async () => {
			const ids = [...new Set(documentIds.map((id) => id.trim()).filter(Boolean))];
			if (ids.length === 0) return 0;
			return this.db.transaction(async (tx) => {
				const placeholders = ids.map((_, index) => `$${index + 2}`).join(", ");
				const result = await tx.query(`DELETE FROM rag_chunks WHERE collection = $1 AND doc_id IN (${placeholders})`, [collection, ...ids]);
				const count = changedRows(result);
				if (count > 0) await this.bumpCollectionRevision(tx, collection, Date.now());
				this.vectorCache.delete(collection);
				return count;
			});
		});
	}

	async deleteCollection(collection: string): Promise<boolean> {
		return this.enqueue(async () => this.db.transaction(async (tx) => {
			const state = await this.readCollectionState(tx, collection);
			const deleted = await tx.query("DELETE FROM rag_chunks WHERE collection = $1", [collection]);
			const count = changedRows(deleted);
			if (state) {
				await tx.query("UPDATE rag_collections SET dimensions = 0, model = '', revision = revision + 1, updated_at = $2 WHERE collection = $1", [collection, Date.now()]);
			}
			this.vectorCache.delete(collection);
			return count > 0;
		}));
	}

	async collections(): Promise<string[]> {
		return this.enqueue(async () => {
			const result = await this.db.query<{ collection: string }>("SELECT DISTINCT collection FROM rag_chunks ORDER BY collection");
			return result.rows.map((row) => row.collection);
		});
	}

	async collectionState(collection: string): Promise<CollectionState | undefined> {
		return this.enqueue(() => this.readCollectionState(this.db, collection));
	}

	async workspaceFingerprint(path: string): Promise<{ fingerprint: string; fileCount: number; chunkCount: number } | undefined> {
		return this.enqueue(async () => {
			const result = await this.db.query<WorkspaceRow>("SELECT fingerprint, file_count, chunk_count FROM workspace_state WHERE path = $1", [path]);
			const row = result.rows[0];
			return row ? {
				fingerprint: row.fingerprint,
				fileCount: numberValue(row.file_count, "workspace file count"),
				chunkCount: numberValue(row.chunk_count, "workspace chunk count"),
			} : undefined;
		});
	}

	async setWorkspaceFingerprint(path: string, fingerprint: string, fileCount: number, chunkCount: number): Promise<void> {
		return this.enqueue(() => this.db.transaction(async (tx) => {
			await tx.query(`
				INSERT INTO workspace_state(path, fingerprint, file_count, chunk_count, indexed_at)
				VALUES ($1, $2, $3, $4, $5)
				ON CONFLICT(path) DO UPDATE SET
					fingerprint = EXCLUDED.fingerprint,
					file_count = EXCLUDED.file_count,
					chunk_count = EXCLUDED.chunk_count,
					indexed_at = EXCLUDED.indexed_at
			`, [path, fingerprint, fileCount, chunkCount, Date.now()]);
		}));
	}

	async searchChunks(collection: string, queryVector: Float32Array, text: string, topK: number, hybrid: boolean): Promise<RagHit[]> {
		return this.enqueue(async () => {
			const bounded = boundedLimit(topK, 1, 1_000);
			const dense = await this.searchDense(collection, queryVector, bounded);
			if (!hybrid) return dense;
			const lexical = await this.searchLexical(collection, text, bounded * 2);
			if (lexical.length === 0) return dense;
			const scores = new Map<string, number>();
			dense.forEach((hit, rank) => scores.set(hitKey(hit), 1 / (60 + rank + 1)));
			const hits = new Map<string, RagHit>();
			for (const hit of dense) hits.set(hitKey(hit), hit);
			const lexicalKeys = new Set<string>();
			for (const [id, rank] of lexical) {
				const hydrated = await this.hydrateById(id);
				if (!hydrated) continue;
				const key = hitKey(hydrated);
				lexicalKeys.add(key);
				scores.set(key, (scores.get(key) ?? 0) + 1 / (60 + rank + 1));
				hits.set(key, hydrated);
			}
			return [...scores.entries()]
				.sort((a, b) => b[1] - a[1] || Number(lexicalKeys.has(b[0])) - Number(lexicalKeys.has(a[0])) || a[0].localeCompare(b[0]))
				.slice(0, bounded)
				.map(([key, score]) => ({ ...hits.get(key)!, score }));
		});
	}

	private async listInternal(scope?: MemoryScope, scopeKey?: string, limit = 200): Promise<MemoryRecord[]> {
		const bounded = boundedLimit(limit, 1, 1_000);
		const where: string[] = [];
		const params: Array<string | number> = [];
		if (scope) {
			params.push(scope);
			where.push(`scope = $${params.length}`);
		}
		if (scopeKey !== undefined) {
			params.push(scopeKey);
			where.push(`scope_key = $${params.length}`);
		}
		const clause = where.length > 0 ? ` WHERE ${where.join(" AND ")}` : "";
		params.push(bounded);
		const result = await this.db.query<MemoryRow>(`SELECT * FROM memories${clause} ORDER BY pinned DESC, updated_at DESC, id LIMIT $${params.length}`, params);
		return result.rows.map(rowToMemory);
	}

	private async reserveCollectionTx(tx: Transaction, collection: string, dimensions: number, model: string, now = Date.now()): Promise<void> {
		if (!collection.trim()) throw new Error("collection cannot be empty");
		if (!model.trim()) throw new Error("embedding model cannot be empty");
		if (!Number.isSafeInteger(dimensions) || dimensions <= 0) throw new EmbeddingShapeError("collection dimensions must be positive");
		await tx.query(`
			INSERT INTO rag_collections(collection, dimensions, model, revision, created_at, updated_at)
			VALUES ($1, $2, $3, 0, $4, $4)
			ON CONFLICT(collection) DO UPDATE SET
				dimensions = CASE WHEN rag_collections.dimensions = 0 THEN EXCLUDED.dimensions ELSE rag_collections.dimensions END,
				model = CASE WHEN rag_collections.dimensions = 0 THEN EXCLUDED.model ELSE rag_collections.model END
		`, [collection, dimensions, model, now]);
		const result = await tx.query<CollectionStateRow>("SELECT dimensions, model FROM rag_collections WHERE collection = $1", [collection]);
		const row = result.rows[0];
		if (!row) throw new Error(`could not reserve collection ${collection}`);
		const existingDimensions = numberValue(row.dimensions, "collection dimensions");
		if (existingDimensions !== dimensions) throw new DimensionMismatchError(existingDimensions, dimensions);
		if (row.model !== model) throw new EmbeddingModelMismatchError(collection, row.model, model);
	}

	private async bumpCollectionRevision(executor: QueryExecutor, collection: string, now: number): Promise<void> {
		await executor.query("UPDATE rag_collections SET revision = revision + 1, updated_at = $2 WHERE collection = $1", [collection, now]);
	}

	private async readCollectionState(executor: QueryExecutor, collection: string): Promise<CollectionState | undefined> {
		const result = await executor.query<CollectionStateRow>("SELECT dimensions, model, revision FROM rag_collections WHERE collection = $1", [collection]);
		const row = result.rows[0];
		return row ? {
			dimensions: numberValue(row.dimensions, "collection dimensions"),
			model: row.model,
			revision: numberValue(row.revision, "collection revision"),
		} : undefined;
	}

	private async searchDense(collection: string, queryVector: Float32Array, topK: number): Promise<RagHit[]> {
		const cache = await this.loadVectorCache(collection);
		if (!cache || cache.chunks.length === 0) return [];
		if (cache.dimensions !== queryVector.length) throw new DimensionMismatchError(cache.dimensions, queryVector.length);
		const query = normalizeVector(queryVector);
		const ranked = cache.chunks.map((chunk) => ({
			chunk,
			score: cosine(query, chunk.vector),
		}));
		ranked.sort((a, b) => b.score - a.score || a.chunk.id.localeCompare(b.chunk.id));
		return ranked.slice(0, topK).map(({ chunk, score }) => ({
			collection,
			documentId: chunk.documentId,
			path: chunk.path,
			content: chunk.content,
			score,
		}));
	}

	private async searchLexical(collection: string, text: string, topK: number): Promise<Array<[string, number]>> {
		const query = text.trim();
		if (!query) return [];
		const ftsQuery = postgresFtsQuery(query);
		if (this.ragFtsAvailable && ftsQuery) {
			try {
				const result = await this.db.query<{ id: string }>(`
					SELECT id
					FROM rag_chunks
					WHERE collection = $1
					  AND to_tsvector('simple', content) @@ to_tsquery('simple', $2)
					ORDER BY ts_rank_cd(to_tsvector('simple', content), to_tsquery('simple', $2)) DESC, updated_at DESC, id
					LIMIT $3
				`, [collection, ftsQuery, topK]);
				return result.rows.map((row, index) => [row.id, index]);
			} catch {
				// Fall through to ILIKE when FTS is unavailable or the query is malformed.
			}
		}
		const result = await this.db.query<{ id: string }>(`
			SELECT id FROM rag_chunks
			WHERE collection = $1 AND content ILIKE $2 ESCAPE '\\'
			ORDER BY updated_at DESC, id
			LIMIT $3
		`, [collection, `%${escapeLike(query.toLowerCase())}%`, topK]);
		return result.rows.map((row, index) => [row.id, index]);
	}

	private async hydrateById(id: string): Promise<RagHit | undefined> {
		const result = await this.db.query<ChunkRow>("SELECT * FROM rag_chunks WHERE id = $1", [id]);
		const row = result.rows[0];
		return row ? {
			collection: row.collection,
			documentId: row.doc_id,
			path: row.path,
			content: row.content,
			score: 0,
		} : undefined;
	}

	private async loadVectorCache(collection: string): Promise<VectorCache | undefined> {
		const state = await this.readCollectionState(this.db, collection);
		if (!state || (state.revision === 0 && state.dimensions === 0)) return undefined;
		const cached = this.vectorCache.get(collection);
		if (cached && cached.revision === state.revision && cached.dimensions === state.dimensions && cached.model === state.model) return cached;
		const result = await this.db.query<ChunkRow>("SELECT * FROM rag_chunks WHERE collection = $1", [collection]);
		const chunks: CachedChunk[] = [];
		for (const row of result.rows) {
			const dimensions = numberValue(row.dimensions, "stored vector dimensions");
			if (dimensions !== state.dimensions) throw new DimensionMismatchError(state.dimensions, dimensions);
			const chunk = rowToChunk(row);
			chunk.vector = normalizeVector(chunk.vector);
			chunks.push(chunk);
		}
		const cache: VectorCache = { revision: state.revision, dimensions: state.dimensions, model: state.model, chunks };
		this.vectorCache.set(collection, cache);
		return cache;
	}
}
