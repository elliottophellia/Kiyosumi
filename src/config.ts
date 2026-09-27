import { createHash } from "node:crypto";
import { homedir } from "node:os";
import { isAbsolute, resolve } from "node:path";
import type { EmbeddingApiStyle, EmbeddingEndpointConfig, RerankApiStyle, RerankEndpointConfig } from "./types.js";

export interface KiyosumiConfig {
	dataDir: string;
	memory: { enabled: boolean; charLimit: number };
	rag: {
		enabled: boolean;
		providerName: string;
		embedding: EmbeddingEndpointConfig;
		rerank: RerankEndpointConfig;
		embeddingDimensions: number;
		embeddingIdentity: string;
		chunkSize: number;
		chunkOverlap: number;
		recall: number;
		topK: number;
		hybrid: boolean;
		rerankEnabled: boolean;
		dedupe: boolean;
		autoContext: boolean;
		requestTimeoutMs: number;
		contextTimeoutMs: number;
		contextMaxChars: number;
		contextMaxPassages: number;
		maxFileBytes: number;
		maxFiles: number;
		maxIndexBytes: number;
	};
}

type Env = Record<string, string | undefined>;
const DEFAULTS = {
	embeddingModel: "voyage-4",
	embeddingDimensions: 1024,
	rerankModel: "rerank-2.5",
	chunkSize: 1200,
	chunkOverlap: 200,
	recall: 40,
	topK: 8,
	memoryCharLimit: 12_000,
	requestTimeoutMs: 60_000,
	contextTimeoutMs: 8_000,
	contextMaxChars: 4_000,
	contextMaxPassages: 6,
	maxFileBytes: 2 * 1024 * 1024,
	maxFiles: 3_000,
	maxIndexBytes: 64 * 1024 * 1024,
} as const;
const EMBEDDING_DIMENSIONS: Record<number, true> = { 256: true, 512: true, 1024: true, 2048: true };
export const DEFAULT_VOYAGE_EMBEDDING_ENDPOINT = "https://api.voyageai.com/v1/embeddings";


function boolEnv(env: Env, name: string, fallback: boolean): boolean {

	const value = env[name]?.trim().toLowerCase();
	if (!value) return fallback;
	if (["1", "true", "yes", "on"].includes(value)) return true;
	if (["0", "false", "no", "off"].includes(value)) return false;
	throw new Error(`${name} must be one of 1, 0, true, false, yes, no, on, or off`);
}

function intEnv(env: Env, name: string, fallback: number, min: number, max: number): number {
	const raw = env[name]?.trim();
	if (!raw) return fallback;
	if (!/^\d+$/.test(raw)) throw new Error(`${name} must be an integer`);
	const value = Number(raw);
	if (!Number.isSafeInteger(value) || value < min || value > max) throw new Error(`${name} must be between ${min} and ${max}`);
	return value;
}

function dataDirectory(env: Env): string {
	const explicit = env.KIYOSUMI_DATA_DIR?.trim();
	if (explicit) return resolve(explicit);
	const agentDir = env.PI_CODING_AGENT_DIR?.trim();
	if (agentDir) return resolve(agentDir, "kiyosumi");
	const profile = env.OMP_PROFILE?.trim();
	if (profile) return resolve(homedir(), ".omp", "profiles", profile, "agent", "kiyosumi");
	return resolve(homedir(), ".omp", "agent", "kiyosumi");
}

function normalizedEndpoint(full: string | undefined, fullName: string, path: string, base: string | undefined): string {
	const raw = full?.trim() || (base?.trim() ? `${base.trim().replace(/\/+$/, "")}${path}` : `https://api.voyageai.com/v1${path}`);
	if (!isAbsolute(raw) && !raw.startsWith("http://") && !raw.startsWith("https://")) throw new Error(`${fullName} must be an absolute URL`);
	return raw.replace(/\/+$/, "");
}
export function hasConfiguredEmbeddingProvider(rag: KiyosumiConfig["rag"]): boolean {
	return rag.embedding.endpoint !== DEFAULT_VOYAGE_EMBEDDING_ENDPOINT || Boolean(rag.embedding.apiKey);
}

export function redactEndpoint(endpoint: string): string {
	try {
		const url = new URL(endpoint);
		url.username = "";
		url.password = "";
		url.search = "";
		url.hash = "";
		return url.toString().replace(/\/$/, "");
	} catch {
		return "configured endpoint";
	}
}

function embeddingStyleEnv(env: Env): EmbeddingApiStyle {
	const value = env.KIYOSUMI_EMBEDDING_API_STYLE?.trim().toLowerCase() || "voyage";
	if (value !== "voyage" && value !== "openai") throw new Error("KIYOSUMI_EMBEDDING_API_STYLE must be voyage or openai");
	return value;
}

function rerankStyleEnv(env: Env): RerankApiStyle {
	const value = env.KIYOSUMI_RERANK_API_STYLE?.trim().toLowerCase() || "voyage";
	if (value !== "voyage" && value !== "generic") throw new Error("KIYOSUMI_RERANK_API_STYLE must be voyage or generic");
	return value;
}

export function loadConfig(env: Env = process.env): KiyosumiConfig {
	const chunkSize = intEnv(env, "KIYOSUMI_CHUNK_SIZE", DEFAULTS.chunkSize, 200, 100_000);
	const defaultOverlap = Math.min(200, Math.max(1, Math.floor(chunkSize / 6)));
	const chunkOverlap = intEnv(env, "KIYOSUMI_CHUNK_OVERLAP", defaultOverlap, 0, chunkSize - 1);
	const topK = intEnv(env, "KIYOSUMI_TOP_K", DEFAULTS.topK, 1, 100);
	const recall = intEnv(env, "KIYOSUMI_RECALL", Math.max(DEFAULTS.recall, topK), topK, 1_000);
	const embeddingDimensions = intEnv(env, "KIYOSUMI_EMBED_DIMENSIONS", DEFAULTS.embeddingDimensions, 256, 2048);
	if (!EMBEDDING_DIMENSIONS[embeddingDimensions]) throw new Error("KIYOSUMI_EMBED_DIMENSIONS must be 256, 512, 1024, or 2048");
	const embeddingStyle = embeddingStyleEnv(env);
	const rerankStyle = rerankStyleEnv(env);
	const embeddingEndpoint = normalizedEndpoint(env.KIYOSUMI_EMBEDDING_ENDPOINT, "KIYOSUMI_EMBEDDING_ENDPOINT", "/embeddings", env.KIYOSUMI_EMBEDDING_BASE_URL ?? env.KIYOSUMI_VOYAGE_BASE_URL);
	const endpointIdentity = createHash("sha256").update(embeddingEndpoint).digest("hex").slice(0, 12);
	const embeddingModel = env.KIYOSUMI_EMBEDDING_MODEL?.trim() || env.KIYOSUMI_EMBED_MODEL?.trim() || DEFAULTS.embeddingModel;
	const embeddingApiKey = env.KIYOSUMI_EMBEDDING_API_KEY?.trim() || env.VOYAGE_API_KEY?.trim() || undefined;
	const rerankEndpoint = normalizedEndpoint(env.KIYOSUMI_RERANK_ENDPOINT, "KIYOSUMI_RERANK_ENDPOINT", "/rerank", env.KIYOSUMI_RERANK_BASE_URL ?? env.KIYOSUMI_EMBEDDING_BASE_URL ?? env.KIYOSUMI_VOYAGE_BASE_URL);
	const rerankModel = env.KIYOSUMI_RERANK_MODEL?.trim() || DEFAULTS.rerankModel;
	const rerankApiKey = env.KIYOSUMI_RERANK_API_KEY?.trim() || env.VOYAGE_API_KEY?.trim() || undefined;
	const providerName = env.KIYOSUMI_PROVIDER_NAME?.trim() || "voyage";
	const embeddingIdentity = `${providerName}:${embeddingStyle}:${endpointIdentity}:${embeddingModel}:${embeddingDimensions}`;

	return {
		dataDir: dataDirectory(env),
		memory: { enabled: boolEnv(env, "KIYOSUMI_MEMORY_ENABLED", true), charLimit: intEnv(env, "KIYOSUMI_MEMORY_CHAR_LIMIT", DEFAULTS.memoryCharLimit, 500, 200_000) },
		rag: {
			enabled: boolEnv(env, "KIYOSUMI_RAG_ENABLED", true),
			providerName,
			embedding: { endpoint: embeddingEndpoint, apiKey: embeddingApiKey, model: embeddingModel, style: embeddingStyle },
			rerank: { endpoint: rerankEndpoint, apiKey: rerankApiKey, model: rerankModel, style: rerankStyle },
			embeddingDimensions,
			embeddingIdentity,
			chunkSize,
			chunkOverlap,
			recall,
			topK,
			hybrid: boolEnv(env, "KIYOSUMI_HYBRID", true),
			rerankEnabled: boolEnv(env, "KIYOSUMI_RERANK", true),
			dedupe: boolEnv(env, "KIYOSUMI_DEDUPE", true),
			autoContext: boolEnv(env, "KIYOSUMI_AUTO_CONTEXT", true),
			requestTimeoutMs: intEnv(env, "KIYOSUMI_REQUEST_TIMEOUT_MS", DEFAULTS.requestTimeoutMs, 1_000, 600_000),
			contextTimeoutMs: intEnv(env, "KIYOSUMI_CONTEXT_TIMEOUT_MS", DEFAULTS.contextTimeoutMs, 500, 120_000),
			contextMaxChars: intEnv(env, "KIYOSUMI_CONTEXT_MAX_CHARS", DEFAULTS.contextMaxChars, 500, 100_000),
			contextMaxPassages: intEnv(env, "KIYOSUMI_CONTEXT_MAX_PASSAGES", DEFAULTS.contextMaxPassages, 1, 50),
			maxFileBytes: intEnv(env, "KIYOSUMI_MAX_FILE_BYTES", DEFAULTS.maxFileBytes, 1_024, 16 * 1024 * 1024),
			maxFiles: intEnv(env, "KIYOSUMI_MAX_FILES", DEFAULTS.maxFiles, 1, 100_000),
			maxIndexBytes: intEnv(env, "KIYOSUMI_MAX_INDEX_BYTES", DEFAULTS.maxIndexBytes, 1_024, 1024 * 1024 * 1024),
		},
	};
}
