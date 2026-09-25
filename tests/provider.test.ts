import { describe, expect, test } from "bun:test";
import { createProviderBundle, ProviderError } from "../src/provider.ts";

function bundle(fetchImpl: typeof fetch, style: "voyage" | "openai" = "openai") {
	return createProviderBundle({
		name: "test-provider",
		embeddingIdentity: "test-provider:openai:https://provider.test/v1/embeddings:embed-model:2",
		embedding: { endpoint: "https://provider.test/v1/embeddings", apiKey: "test-key", model: "embed-model", style },
		rerank: { endpoint: "https://provider.test/v1/rerank", apiKey: "test-key", model: "rerank-model", style: "generic" },
		embeddingDimensions: 2,
		timeoutMs: 5_000,
		fetchImpl,
	});
}

describe("provider bundle", () => {
	test("supports OpenAI-compatible embedding responses", async () => {
		let requestBody = "";
		let requestUrl = "";
		const fetchImpl: typeof fetch = async (input, init) => {
			requestUrl = String(input);
			requestBody = String(init?.body ?? "");
			return new Response(JSON.stringify({ data: [{ index: 0, embedding: [1, 0] }] }), { status: 200 });
		};
		const vectors = await bundle(fetchImpl).embedding.embedDocuments(["hello"]);
		expect(Array.from(vectors[0]!)).toEqual([1, 0]);
		expect(JSON.parse(requestBody)).toMatchObject({ model: "embed-model", encoding_format: "float" });
		expect(requestUrl).toBe("https://provider.test/v1/embeddings");
	});

	test("keeps Voyage input types when selected", async () => {
		let requestBody = "";
		const fetchImpl: typeof fetch = async (_input, init) => {
			requestBody = String(init?.body ?? "");
			return new Response(JSON.stringify({ data: [{ index: 0, embedding: [1, 0] }] }), { status: 200 });
		};
		await bundle(fetchImpl, "voyage").embedding.embedQuery("question");
		expect(JSON.parse(requestBody).input_type).toBe("query");
	});

	test("parses generic rerank scores and skips malformed items", async () => {
		const fetchImpl: typeof fetch = async () => new Response(JSON.stringify({
			data: [null, { index: 1, score: 0.8 }, { index: 0, relevance_score: 0.9 }],
		}), { status: 200 });
		const hits = await bundle(fetchImpl).rerank.rerank("query", ["a", "b"], 2);
		expect(hits).toEqual([{ index: 0, score: 0.9 }, { index: 1, score: 0.8 }]);
	});

	test("bounds provider error bodies", async () => {
		const fetchImpl: typeof fetch = async () => new Response("x".repeat(1_000_000), { status: 429 });
		try {
			await bundle(fetchImpl).embedding.embedQuery("question");
			throw new Error("expected provider request to fail");
		} catch (error) {
			expect(error).toBeInstanceOf(ProviderError);
			expect((error as ProviderError).message.length).toBeLessThan(350);
		}
	});
});
