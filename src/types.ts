export type MemoryScope = "global" | "project" | "session";
export type EmbeddingApiStyle = "voyage" | "openai";
export type RerankApiStyle = "voyage" | "generic";

export interface ProviderEndpointConfig<TStyle extends string> {
	endpoint: string;
	apiKey: string | undefined;
	model: string;
	style: TStyle;
}

export type EmbeddingEndpointConfig = ProviderEndpointConfig<EmbeddingApiStyle>;
export type RerankEndpointConfig = ProviderEndpointConfig<RerankApiStyle>;

export interface MemoryRecord {
	id: string;
	scope: MemoryScope;
	scopeKey: string;
	key: string;
	content: string;
	source: string;
	pinned: boolean;
	createdAt: number;
	updatedAt: number;
}

export interface SaveMemoryInput {
	scope: MemoryScope;
	scopeKey: string;
	key: string;
	content: string;
	source?: string;
}

export interface RagDocument {
	id: string;
	path: string;
	content: string;
	meta?: Record<string, unknown>;
}

export interface RagChunk {
	id: string;
	collection: string;
	documentId: string;
	path: string;
	index: number;
	content: string;
	embedding: Float32Array;
	meta: Record<string, unknown>;
}

export interface RagHit {
	collection: string;
	documentId: string;
	path: string;
	content: string;
	score: number;
}

export interface RerankHit {
	index: number;
	score: number;
}

export interface EmbeddingProvider {
	embedDocuments(texts: readonly string[], signal?: AbortSignal): Promise<Float32Array[]>;
	embedQuery(text: string, signal?: AbortSignal): Promise<Float32Array>;
}

export interface RerankProvider {
	rerank(query: string, documents: readonly string[], topK: number, signal?: AbortSignal): Promise<RerankHit[]>;
}

export interface ProviderBundle {
	name: string;
	embeddingIdentity: string;
	embedding: EmbeddingProvider;
	rerank: RerankProvider;
}

export interface MemoryRepository {
	save(input: SaveMemoryInput): Promise<MemoryRecord>;
	list(scope?: MemoryScope, scopeKey?: string, limit?: number): Promise<MemoryRecord[]>;
	search(query: string, limit?: number): Promise<MemoryRecord[]>;
	delete(id: string): Promise<boolean>;
	get(id: string): Promise<MemoryRecord | undefined>;
}

export interface SearchRequest {
	collection: string;
	query: string;
	topK?: number;
	signal?: AbortSignal;
}

export interface IndexRequest {
	collection: string;
	documents: readonly RagDocument[];
	signal?: AbortSignal;
	replaceCollection?: boolean;
}

export interface RagEngine {
	search(request: SearchRequest): Promise<RagHit[]>;
	index(request: IndexRequest): Promise<number>;
	collections(): Promise<string[]>;
	deleteCollection(collection: string): Promise<boolean>;
	probe(): Promise<{ ok: boolean; detail: string }>;
}
