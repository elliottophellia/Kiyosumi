import { describe, expect, test } from "bun:test";
import { hasConfiguredEmbeddingProvider, loadConfig, redactEndpoint } from "../src/config.ts";

describe("Kiyosumi configuration", () => {
	test("uses Voyage-compatible defaults", () => {
		const config = loadConfig({});
		expect(config.rag.enabled).toBe(true);
		expect(config.rag.providerName).toBe("voyage");
		expect(config.rag.embedding.model).toBe("voyage-4");
		expect(config.rag.embedding.endpoint).toBe("https://api.voyageai.com/v1/embeddings");
		expect(config.rag.embedding.style).toBe("voyage");
		expect(config.rag.rerank.model).toBe("rerank-2.5");
		expect(config.rag.rerank.endpoint).toBe("https://api.voyageai.com/v1/rerank");
		expect(config.rag.chunkSize).toBe(1200);
		expect(config.rag.topK).toBe(8);
		expect(config.rag.recall).toBe(40);
		expect(config.rag.hybrid).toBe(true);
		expect(config.rag.autoContext).toBe(true);
		expect(config.rag.embedding.apiKey).toBeUndefined();
	});

	test("accepts custom embedding and rerank endpoints", () => {
		const config = loadConfig({
			OMP_PROFILE: "work",
			KIYOSUMI_PROVIDER_NAME: "acme",
			KIYOSUMI_EMBEDDING_BASE_URL: "https://embed.example/v1",
			KIYOSUMI_EMBEDDING_API_KEY: "embed-secret",
			KIYOSUMI_EMBEDDING_MODEL: "acme-embed",
			KIYOSUMI_EMBEDDING_API_STYLE: "openai",
			KIYOSUMI_RERANK_BASE_URL: "https://rerank.example/v1",
			KIYOSUMI_RERANK_API_KEY: "rerank-secret",
			KIYOSUMI_RERANK_MODEL: "acme-rerank",
			KIYOSUMI_RERANK_API_STYLE: "generic",
			KIYOSUMI_TOP_K: "12",
			KIYOSUMI_RECALL: "50",
			KIYOSUMI_CHUNK_SIZE: "800",
			KIYOSUMI_CHUNK_OVERLAP: "80",
			KIYOSUMI_EMBED_DIMENSIONS: "512",
		});
		expect(config.dataDir).toEndWith("/.omp/profiles/work/agent/kiyosumi");
		expect(config.rag.embedding.endpoint).toBe("https://embed.example/v1/embeddings");
		expect(config.rag.embedding.apiKey).toBe("embed-secret");
		expect(config.rag.embeddingIdentity).toMatch(/^acme:openai:[0-9a-f]{12}:acme-embed:512$/);
		expect(config.rag.rerank.apiKey).toBe("rerank-secret");
		expect(config.rag.rerank.style).toBe("generic");
		expect(config.rag.rerank.endpoint).toBe("https://rerank.example/v1/rerank");
		expect(config.rag.topK).toBe(12);
		expect(config.rag.recall).toBe(50);
		expect(config.rag.chunkSize).toBe(800);
		expect(config.rag.chunkOverlap).toBe(80);
		expect(config.rag.embeddingDimensions).toBe(512);
	});

	test("falls back to VOYAGE_API_KEY for the default endpoints", () => {
		const config = loadConfig({ VOYAGE_API_KEY: "secret" });
		expect(config.rag.embedding.apiKey).toBe("secret");
		expect(config.rag.rerank.apiKey).toBe("secret");
	});

	test("redacts endpoint credentials and query strings", () => {
		expect(redactEndpoint("https://user:secret@example.com/v1/embeddings?token=hidden")).toBe("https://example.com/v1/embeddings");
	});

	test("requires a credential only for the default Voyage endpoint", () => {
		expect(hasConfiguredEmbeddingProvider(loadConfig({}).rag)).toBe(false);
		expect(hasConfiguredEmbeddingProvider(loadConfig({ VOYAGE_API_KEY: "secret" }).rag)).toBe(true);
		expect(hasConfiguredEmbeddingProvider(loadConfig({ KIYOSUMI_EMBEDDING_ENDPOINT: "https://embed.example/v1/embeddings" }).rag)).toBe(true);
	});

	test("rejects malformed environment values", () => {
		expect(() => loadConfig({ KIYOSUMI_RAG_ENABLED: "sometimes" })).toThrow("KIYOSUMI_RAG_ENABLED");
		expect(() => loadConfig({ KIYOSUMI_TOP_K: "0" })).toThrow("KIYOSUMI_TOP_K");
		expect(() => loadConfig({ KIYOSUMI_CHUNK_SIZE: "200", KIYOSUMI_CHUNK_OVERLAP: "200" })).toThrow("KIYOSUMI_CHUNK_OVERLAP");
		expect(() => loadConfig({ KIYOSUMI_EMBED_DIMENSIONS: "777" })).toThrow("KIYOSUMI_EMBED_DIMENSIONS");
		expect(() => loadConfig({ KIYOSUMI_EMBEDDING_API_STYLE: "other" })).toThrow("KIYOSUMI_EMBEDDING_API_STYLE");
		expect(() => loadConfig({ KIYOSUMI_RERANK_API_STYLE: "other" })).toThrow("KIYOSUMI_RERANK_API_STYLE");
	});
});
