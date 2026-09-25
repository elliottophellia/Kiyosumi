import { createHash } from "node:crypto";
import { basename, resolve } from "node:path";
import { chunkId, chunkText } from "./chunk.js";
import type { KiyosumiConfig } from "./config.js";
import type { KiyosumiStore } from "./db.js";
import type { IndexRequest, ProviderBundle, RagChunk, RagDocument, RagEngine, RagHit, SearchRequest } from "./types.js";

export const CONVERSATION_COLLECTION = "kiyosumi-conversations";
export const DEFAULT_COLLECTION = "kiyosumi";

interface DocumentParts {
	documentId: string;
	path: string;
	chunks: string[];
	meta: Record<string, unknown>;
}

function collectionName(path: string): string {
	const absolute = resolve(path);
	const digest = createHash("sha256").update(absolute).digest("hex").slice(0, 8);
	const folder = basename(absolute).toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40) || "project";
	return `${folder}-${digest}`;
}

function validCollection(value: string): string {
	const collection = value.trim();
	if (!collection || collection.length > 128 || /[\u0000-\u001f]/.test(collection)) throw new Error("collection must be 1-128 printable characters");
	return collection;
}

function documentParts(document: RagDocument, config: KiyosumiConfig): DocumentParts {
	const documentId = document.id.trim() || document.path.trim();
	if (!documentId) throw new Error("every indexed document needs an id or path");
	const path = document.path.trim() || documentId;
	const chunks = chunkText(document.content, config.rag.chunkSize, config.rag.chunkOverlap);
	return { documentId, path, chunks, meta: { ...(document.meta ?? {}), path } };
}

export class NativeRagEngine implements RagEngine {
	constructor(
		private readonly store: KiyosumiStore,
		private readonly config: KiyosumiConfig,
		private readonly providers: ProviderBundle | undefined,
	) {}

	get configured(): boolean {
		return this.config.rag.enabled && this.providers !== undefined;
	}

	async index(request: IndexRequest): Promise<number> {
		if (!this.config.rag.enabled) throw new Error("RAG is disabled by KIYOSUMI_RAG_ENABLED=0");
		const collection = validCollection(request.collection);
		if (request.documents.length === 0) {
			if (request.replaceCollection) await this.store.deleteCollection(collection);
			return 0;
		}
		if (!this.providers) throw new Error("RAG needs an embedding provider before it can index documents");
		const byId = new Map<string, DocumentParts>();
		for (const document of request.documents) {
			const parts = documentParts(document, this.config);
			byId.set(parts.documentId, parts);
		}
		const pending: Array<{ chunk: Omit<RagChunk, "embedding">; text: string }> = [];
		for (const parts of byId.values()) {
			for (let index = 0; index < parts.chunks.length; index++) {
				const text = parts.chunks[index]!;
				pending.push({
					chunk: {
						id: chunkId(collection, parts.documentId, index),
						collection,
						documentId: parts.documentId,
						path: parts.path,
						index,
						content: text,
						meta: parts.meta,
					},
					text,
				});
			}
		}
		if (request.signal?.aborted) throw new Error("indexing cancelled");
		const vectors = await this.providers.embedding.embedDocuments(pending.map((item) => item.text), request.signal);
		if (vectors.length !== pending.length) throw new Error(`embedding provider returned ${vectors.length} vectors for ${pending.length} chunks`);
		const chunks = pending.map((item, index) => ({ ...item.chunk, embedding: vectors[index]! }));
		await this.store.replaceDocuments(collection, chunks, this.providers.embeddingIdentity, {
			documentIds: [...byId.keys()],
			...(request.replaceCollection === undefined ? {} : { replaceCollection: request.replaceCollection }),
		});
		return chunks.length;
	}

	async search(request: SearchRequest): Promise<RagHit[]> {
		if (!this.config.rag.enabled) throw new Error("RAG is disabled by KIYOSUMI_RAG_ENABLED=0");
		if (!this.providers) throw new Error("RAG needs an embedding provider before it can search");
		const collection = validCollection(request.collection);
		const query = request.query.trim();
		if (!query) throw new Error("query cannot be empty");
		if (request.signal?.aborted) throw new Error("search cancelled");
		await this.store.assertEmbeddingModel(collection, this.providers.embeddingIdentity);
		const topK = Math.min(Math.max(request.topK ?? this.config.rag.topK, 1), 100);
		const recall = Math.max(this.config.rag.recall, topK);
		const vector = await this.providers.embedding.embedQuery(query, request.signal);
		const recalled = await this.store.searchChunks(collection, vector, query, recall, this.config.rag.hybrid);
		return this.rank(query, recalled, topK, request.signal);
	}

	async searchMany(query: string, collections: readonly string[], topK: number, signal?: AbortSignal): Promise<RagHit[]> {
		if (!this.config.rag.enabled || !this.providers) return [];
		const normalizedQuery = query.trim();
		if (!normalizedQuery || collections.length === 0) return [];
		const boundedTopK = Math.min(Math.max(topK, 1), 100);
		const recall = Math.max(this.config.rag.recall, boundedTopK);
		for (const collection of collections) await this.store.assertEmbeddingModel(validCollection(collection), this.providers.embeddingIdentity);
		const vector = await this.providers.embedding.embedQuery(normalizedQuery, signal);
		const recalled: RagHit[] = [];
		for (const collection of collections) {
			recalled.push(...await this.store.searchChunks(validCollection(collection), vector, normalizedQuery, recall, this.config.rag.hybrid));
		}
		return this.rank(normalizedQuery, recalled, boundedTopK, signal);
	}

	private async rank(query: string, recalled: RagHit[], topK: number, signal?: AbortSignal): Promise<RagHit[]> {
		let hits = recalled;
		if (this.config.rag.rerankEnabled && recalled.length > 1 && this.providers) {
			try {
				const reranked = await this.providers.rerank.rerank(query, recalled.map((hit) => hit.content), recalled.length, signal);
				if (reranked.length > 0) {
					hits = reranked.flatMap((ranked) => {
						const hit = recalled[ranked.index];
						return hit ? [{ ...hit, score: ranked.score }] : [];
					});
				}
			} catch (error) {
				if (signal?.aborted) throw error;
			}
		}
		if (this.config.rag.dedupe) {
			const seen = new Set<string>();
			hits = hits.filter((hit) => {
				const key = hit.content.trim();
				if (!key) return true;
				if (seen.has(key)) return false;
				seen.add(key);
				return true;
			});
		}
		return hits.slice(0, topK);
	}

	async collections(): Promise<string[]> {
		return this.store.collections();
	}

	async deleteCollection(collection: string): Promise<boolean> {
		return this.store.deleteCollection(validCollection(collection));
	}

	async probe(): Promise<{ ok: boolean; detail: string }> {
		if (!this.config.rag.enabled) return { ok: false, detail: "RAG is disabled" };
		if (!this.providers) return { ok: false, detail: "No embedding provider is configured" };
		try {
			await this.providers.embedding.embedDocuments(["Kiyosumi health check"]);
			return { ok: true, detail: `${this.providers.name} is reachable` };
		} catch (error) {
			return { ok: false, detail: error instanceof Error ? error.message : String(error) };
		}
	}

	conversationDocument(sessionId: string, userText: string, assistantText: string, title = ""): RagDocument {
		const content = [userText ? `User: ${userText}` : "", assistantText ? `Assistant: ${assistantText}` : ""].filter(Boolean).join("\n\n");
		return {
			id: `conversation:${sessionId}:${createHash("sha256").update(`${userText}\n${assistantText}`).digest("hex").slice(0, 20)}`,
			path: `conversations/${sessionId}`,
			content,
			meta: { kind: "conversation", session_id: sessionId, title },
		};
	}

	projectCollection(projectRoot: string): string {
		return collectionName(projectRoot);
	}
}

export function messageText(message: unknown): string {
	if (!message || typeof message !== "object") return "";
	const record = message as Record<string, unknown>;
	if (typeof record.content === "string") return record.content;
	if (!Array.isArray(record.content)) return "";
	return record.content
		.filter((part): part is Record<string, unknown> => Boolean(part) && typeof part === "object")
		.filter((part) => part.type === "text" && typeof part.text === "string")
		.map((part) => part.text as string)
		.join("\n")
		.trim();
}
