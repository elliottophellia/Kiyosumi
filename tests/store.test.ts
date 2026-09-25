import { describe, expect, test } from "bun:test";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadConfig } from "../src/config.ts";
import { DimensionMismatchError, EmbeddingModelMismatchError, KiyosumiStore } from "../src/db.ts";
import { NativeRagEngine } from "../src/rag.ts";
import type { ProviderBundle, RagDocument } from "../src/types.ts";

function settings(overrides: Record<string, string> = {}) {
	return loadConfig({ KIYOSUMI_EMBED_DIMENSIONS: "256", ...overrides });
}

function vector(value: number): Float32Array {
	const result = new Float32Array(256);
	result[0] = value;
	return result;
}

function providers(overrides: Partial<ProviderBundle> = {}): ProviderBundle {
	return {
		name: "test-provider",
		embeddingIdentity: "test-provider:embed-model:256",
		embedding: {
			embedDocuments: async (texts) => texts.map((text) => text.includes("kubernetes") || text.includes("old") ? vector(0.1) : vector(1)),
			embedQuery: async (text) => text.includes("kubernetes") ? vector(0.1) : vector(1),
		},
		rerank: {
			rerank: async (_query, documents) => documents.map((_document, index) => ({ index, score: documents.length - index })),
		},
		...overrides,
	};
}

async function newStore(): Promise<{ dir: string; store: KiyosumiStore }> {
	const dir = await mkdtemp(join(tmpdir(), "kiyosumi-store-"));
	return { dir, store: await KiyosumiStore.open(dir) };
}

describe("Kiyosumi memory", () => {
	test("upserts a keyed fact and finds its replacement", async () => {
		const { store } = await newStore();
		await store.save({ scope: "global", scopeKey: "", key: "language", content: "User writes in English", source: "test" });
		await store.save({ scope: "global", scopeKey: "", key: "language", content: "User writes in Japanese and English", source: "test" });
		const found = await store.search("Japanese", 10);
		expect(found).toHaveLength(1);
		expect(found[0]!.content).toContain("Japanese");
		expect(found[0]!.key).toBe("language");
		await store.close();
	});

	test("deletes a memory by its stable id", async () => {
		const { store } = await newStore();
		const memory = await store.save({ scope: "project", scopeKey: "/tmp/project", key: "database", content: "Project uses PostgreSQL" });
		expect(await store.delete(memory.id)).toBe(true);
		expect(await store.get(memory.id)).toBeUndefined();
		expect(await store.delete(memory.id)).toBe(false);
		await store.close();
	});
});

describe("Kiyosumi retrieval", () => {
	test("does not embed empty multi-collection searches", async () => {
		const { store } = await newStore();
		let embedCalls = 0;
		const bundle = providers({ embedding: { embedDocuments: async () => [], embedQuery: async () => { embedCalls++; return vector(1); } }, rerank: { rerank: async () => [] } });
		const engine = new NativeRagEngine(store, settings(), bundle);
		expect(await engine.searchMany("", ["missing"], 8)).toEqual([]);
		expect(await engine.searchMany("query", [], 8)).toEqual([]);
		expect(embedCalls).toBe(0);
		await store.close();
	});

	test("indexes, searches, reranks, and deduplicates passages", async () => {
		const { store } = await newStore();
		const engine = new NativeRagEngine(store, settings({ KIYOSUMI_RERANK: "1" }), providers());
		const documents: RagDocument[] = [
			{ id: "a", path: "a.md", content: "The project uses a blue lantern." },
			{ id: "b", path: "b.md", content: "The project uses a blue lantern." },
		];
		expect(await engine.index({ collection: "notes", documents })).toBe(2);
		const hits = await engine.search({ collection: "notes", query: "blue lantern", topK: 2 });
		expect(hits).toHaveLength(1);
		expect(hits[0]!.path).toBe("a.md");
		await store.close();
	});

	test("hybrid lexical recall rescues an orthogonal dense miss", async () => {
		const { store } = await newStore();
		const engine = new NativeRagEngine(store, settings({ KIYOSUMI_HYBRID: "1", KIYOSUMI_RERANK: "0" }), providers());
		await engine.index({ collection: "hybrid", documents: [
			{ id: "dense", path: "dense.md", content: "ordinary application notes" },
			{ id: "lexical", path: "lexical.md", content: "kubernetes scheduler internals" },
		] });
		const hits = await engine.search({ collection: "hybrid", query: "kubernetes", topK: 1 });
		expect(hits[0]!.path).toBe("lexical.md");
		await store.close();
	});

	test("rejects a second vector dimensionality for a collection", async () => {
		const { store } = await newStore();
		const first = vector(1);
		const wrong = new Float32Array([1, 0]);
		await store.replaceDocuments("shapes", [{ id: "one", collection: "shapes", documentId: "doc", path: "doc", index: 0, content: "one", embedding: first, meta: {} }], "test-provider:embed-model:256");
		await expect(store.replaceDocuments("shapes", [{ id: "two", collection: "shapes", documentId: "doc", path: "doc", index: 0, content: "two", embedding: wrong, meta: {} }], "test-provider:embed-model:256")).rejects.toBeInstanceOf(DimensionMismatchError);
		await store.close();
	});

	test("rejects a model change for an existing collection", async () => {
		const { store } = await newStore();
		const first = vector(1);
		await store.replaceDocuments("models", [{ id: "one", collection: "models", documentId: "doc", path: "doc", index: 0, content: "one", embedding: first, meta: {} }], "voyage-4");
		await expect(store.replaceDocuments("models", [{ id: "two", collection: "models", documentId: "doc", path: "doc", index: 0, content: "two", embedding: first, meta: {} }], "voyage-4-large")).rejects.toBeInstanceOf(EmbeddingModelMismatchError);
		await store.close();
	});

	test("allows a new vector shape after deleting an empty collection", async () => {
		const { store } = await newStore();
		const first = vector(1);
		await store.replaceDocuments("reset", [{ id: "one", collection: "reset", documentId: "doc", path: "doc", index: 0, content: "one", embedding: first, meta: {} }], "voyage-4");
		await store.deleteCollection("reset");
		const second = new Float32Array([1, 0]);
		await store.replaceDocuments("reset", [{ id: "two", collection: "reset", documentId: "doc", path: "doc", index: 0, content: "two", embedding: second, meta: {} }], "voyage-4-large");
		expect((await store.collectionState("reset"))?.dimensions).toBe(2);
		await store.close();
	});

	test("reindexing a shorter document removes stale tail chunks", async () => {
		const { store } = await newStore();
		const config = settings({ KIYOSUMI_CHUNK_SIZE: "200", KIYOSUMI_CHUNK_OVERLAP: "20", KIYOSUMI_RERANK: "0" });
		const engine = new NativeRagEngine(store, config, providers());
		await engine.index({ collection: "tail", documents: [{ id: "doc", path: "doc.txt", content: `${"old ".repeat(160)}` }] });
		await engine.index({ collection: "tail", documents: [{ id: "doc", path: "doc.txt", content: "new passage" }] });
		const hits = await engine.search({ collection: "tail", query: "new passage", topK: 20 });
		expect(hits.length).toBeGreaterThan(0);
		expect(hits.some((hit) => hit.content.includes("old"))).toBe(false);
		await store.close();
	});

	test("sees writes made before a reopened store loads its vectors", async () => {
		const first = await newStore();
		const config = settings({ KIYOSUMI_RERANK: "0" });
		const firstEngine = new NativeRagEngine(first.store, config, providers());
		await firstEngine.index({ collection: "reopen", documents: [{ id: "one", path: "one.md", content: "first document" }] });
		await firstEngine.search({ collection: "reopen", query: "first" });
		await first.store.close();
		const secondStore = await KiyosumiStore.open(first.dir);
		const secondEngine = new NativeRagEngine(secondStore, config, providers());
		const hits = await secondEngine.search({ collection: "reopen", query: "first", topK: 1 });
		expect(hits[0]!.path).toBe("one.md");
		await secondStore.close();
	});
});
