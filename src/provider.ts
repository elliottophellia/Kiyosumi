import type {
	EmbeddingEndpointConfig,
	EmbeddingProvider,
	ProviderBundle,
	RerankEndpointConfig,
	RerankProvider,
	RerankHit,
} from "./types.js";

export interface ProviderBundleOptions {
	name: string;
	embeddingIdentity: string;
	embedding: EmbeddingEndpointConfig;
	rerank: RerankEndpointConfig;
	embeddingDimensions: number;
	timeoutMs: number;
	fetchImpl?: typeof fetch;
}

export class ProviderError extends Error {
	constructor(message: string, readonly status?: number) {
		super(message);
		this.name = "ProviderError";
	}
}

function requestSignal(timeoutMs: number, signal?: AbortSignal): { signal: AbortSignal; cancel: () => void } {
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(new Error(`provider request timed out after ${timeoutMs}ms`)), timeoutMs);
	const onAbort = () => controller.abort(signal?.reason);
	if (signal) {
		if (signal.aborted) onAbort();
		else signal.addEventListener("abort", onAbort, { once: true });
	}
	return {
		signal: controller.signal,
		cancel: () => {
			clearTimeout(timer);
			signal?.removeEventListener("abort", onAbort);
		},
	};
}

async function responseError(response: Response): Promise<ProviderError> {
	const reader = response.body?.getReader();
	if (!reader) return new ProviderError(`provider returned ${response.status}`, response.status);
	const decoder = new TextDecoder();
	let body = "";
	let bytesRead = 0;
	try {
		while (bytesRead < 300) {
			const result = await reader.read();
			if (result.done) {
				body += decoder.decode();
				break;
			}
			const remaining = 300 - bytesRead;
			const chunk = result.value.subarray(0, remaining);
			bytesRead += chunk.byteLength;
			body += decoder.decode(chunk, { stream: true });
			if (result.value.byteLength > remaining) {
				try {
					await reader.cancel();
				} catch {
					// The response is already bounded; cancellation is best effort.
				}
				break;
			}
		}
	} finally {
		reader.releaseLock();
	}
	return new ProviderError(`provider returned ${response.status}: ${body}`, response.status);
}


function parseVector(value: unknown, expectedDimensions: number): Float32Array {
	if (!Array.isArray(value) || value.length !== expectedDimensions) throw new ProviderError("provider returned a vector with an unexpected shape");
	const vector = new Float32Array(expectedDimensions);
	for (let index = 0; index < expectedDimensions; index++) {
		const component = value[index];
		if (typeof component !== "number" || !Number.isFinite(component)) throw new ProviderError(`provider returned a non-finite vector component at index ${index}`);
		vector[index] = component;
	}
	return vector;
}

class HttpEmbeddingProvider implements EmbeddingProvider {
	constructor(
		private readonly endpointConfig: EmbeddingEndpointConfig,
		private readonly dimensions: number,
		private readonly timeoutMs: number,
		private readonly fetchImpl: typeof fetch,
	) {}

	async embedDocuments(texts: readonly string[], signal?: AbortSignal): Promise<Float32Array[]> {
		if (texts.length === 0) return [];
		const output: Float32Array[] = [];
		for (let start = 0; start < texts.length; start += 128) {
			output.push(...await this.embed(texts.slice(start, start + 128), "document", signal));
		}
		return output;
	}

	async embedQuery(text: string, signal?: AbortSignal): Promise<Float32Array> {
		const [vector] = await this.embed([text], "query", signal);
		if (!vector) throw new ProviderError("provider returned no query embedding");
		return vector;
	}

	private async embed(texts: readonly string[], inputType: "query" | "document", signal?: AbortSignal): Promise<Float32Array[]> {
		const body = this.endpointConfig.style === "voyage"
			? { input: texts, model: this.endpointConfig.model, input_type: inputType, output_dimension: this.dimensions, truncation: true }
			: { input: texts, model: this.endpointConfig.model, encoding_format: "float" };
		const parsed = await this.post(this.endpointConfig.endpoint, body, signal);
		const rawItems = Array.isArray(parsed.data) ? parsed.data : [];
		const vectors: Array<Float32Array | undefined> = new Array(texts.length);
		for (const raw of rawItems) {
			if (!raw || typeof raw !== "object") continue;
			const item = raw as Record<string, unknown>;
			if (typeof item.index !== "number" || !Number.isInteger(item.index) || item.index < 0 || item.index >= texts.length || vectors[item.index]) continue;
			vectors[item.index] = parseVector(item.embedding, this.dimensions);
		}
		if (vectors.some((vector) => !vector)) throw new ProviderError("provider omitted one or more embedding vectors");
		return vectors as Float32Array[];
	}

	private async post(url: string, body: unknown, signal?: AbortSignal): Promise<Record<string, unknown>> {
		const request = requestSignal(this.timeoutMs, signal);
		try {
			const headers: Record<string, string> = { "Content-Type": "application/json", "User-Agent": "Kiyosumi/0.1" };
			if (this.endpointConfig.apiKey) headers.Authorization = `Bearer ${this.endpointConfig.apiKey}`;
			const response = await this.fetchImpl(url, { method: "POST", headers, body: JSON.stringify(body), signal: request.signal });
			if (!response.ok) throw await responseError(response);
			const parsed: unknown = await response.json();
			if (!parsed || typeof parsed !== "object") throw new ProviderError("provider response was not an object");
			return parsed as Record<string, unknown>;
		} catch (error) {
			if (error instanceof ProviderError) throw error;
			if (request.signal.aborted) throw new ProviderError(`provider request aborted: ${String(request.signal.reason ?? "aborted")}`);
			throw new ProviderError(`provider request failed: ${error instanceof Error ? error.message : String(error)}`);
		} finally {
			request.cancel();
		}
	}
}

class HttpRerankProvider implements RerankProvider {
	constructor(private readonly endpointConfig: RerankEndpointConfig, private readonly timeoutMs: number, private readonly fetchImpl: typeof fetch) {}

	async rerank(query: string, documents: readonly string[], topK: number, signal?: AbortSignal): Promise<RerankHit[]> {
		if (documents.length === 0) return [];
		const body = this.endpointConfig.style === "voyage"
			? { query, documents, model: this.endpointConfig.model, top_k: Math.min(Math.max(topK, 1), documents.length), return_documents: false, truncation: true }
			: { query, documents, model: this.endpointConfig.model, top_n: Math.min(Math.max(topK, 1), documents.length) };
		const request = requestSignal(this.timeoutMs, signal);
		try {
			const headers: Record<string, string> = { "Content-Type": "application/json", "User-Agent": "Kiyosumi/0.1" };
			if (this.endpointConfig.apiKey) headers.Authorization = `Bearer ${this.endpointConfig.apiKey}`;
			const response = await this.fetchImpl(this.endpointConfig.endpoint, { method: "POST", headers, body: JSON.stringify(body), signal: request.signal });
			if (!response.ok) throw await responseError(response);
			const parsed: unknown = await response.json();
			if (!parsed || typeof parsed !== "object") throw new ProviderError("provider rerank response was not an object");
			const record = parsed as Record<string, unknown>;
			const rawItems = Array.isArray(record.data) ? record.data : Array.isArray(record.results) ? record.results : [];
			const hits: RerankHit[] = [];
			const seen = new Set<number>();
			for (const raw of rawItems) {
				if (!raw || typeof raw !== "object") continue;
				const item = raw as Record<string, unknown>;
				const score = typeof item.relevance_score === "number" ? item.relevance_score : item.score;
				if (typeof item.index !== "number" || !Number.isInteger(item.index) || item.index < 0 || item.index >= documents.length || seen.has(item.index)) continue;
				if (typeof score !== "number" || !Number.isFinite(score)) continue;
				seen.add(item.index);
				hits.push({ index: item.index, score });
			}
			hits.sort((left, right) => right.score - left.score || left.index - right.index);
			return hits.slice(0, topK);
		} catch (error) {
			if (error instanceof ProviderError) throw error;
			if (request.signal.aborted) throw new ProviderError(`provider rerank request aborted: ${String(request.signal.reason ?? "aborted")}`);
			throw new ProviderError(`provider rerank request failed: ${error instanceof Error ? error.message : String(error)}`);
		} finally {
			request.cancel();
		}
	}
}

export function createProviderBundle(options: ProviderBundleOptions): ProviderBundle {
	const fetchImpl = options.fetchImpl ?? fetch;
	return {
		name: options.name,
		embeddingIdentity: options.embeddingIdentity,
		embedding: new HttpEmbeddingProvider(options.embedding, options.embeddingDimensions, options.timeoutMs, fetchImpl),
		rerank: new HttpRerankProvider(options.rerank, options.timeoutMs, fetchImpl),
	};
}
