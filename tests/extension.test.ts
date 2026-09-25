import { describe, expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@oh-my-pi/pi-coding-agent";
import kiyosumiExtension from "../src/extension.ts";

function schema(): Record<string, unknown> {
	const chain: Record<string, unknown> = {};
	for (const method of ["optional", "int", "min", "max", "describe"]) {
		chain[method] = () => chain;
	}
	return chain;
}

describe("Kiyosumi extension lifecycle", () => {
	test.serial("injects durable memory even when no retrieval collection exists", async () => {
		const dataDir = await mkdtemp(join(tmpdir(), "kiyosumi-extension-"));
		const previous = {
			data: process.env.KIYOSUMI_DATA_DIR,
			autoContext: process.env.KIYOSUMI_AUTO_CONTEXT,
			voyage: process.env.VOYAGE_API_KEY,
		};
		process.env.KIYOSUMI_DATA_DIR = dataDir;
		process.env.KIYOSUMI_AUTO_CONTEXT = "1";
		process.env.VOYAGE_API_KEY = "test-key";
		const handlers = new Map<string, (...args: never[]) => unknown>();
		const commands = new Map<string, { handler: (args: string, ctx: ExtensionContext) => Promise<void> | void }>();
		const z = { object: schema, string: schema, number: schema, enum: schema };
		const pi = {
			setLabel: () => undefined,
			zod: z,
			logger: { warn: () => undefined, debug: () => undefined },
			on: (event: string, handler: (...args: never[]) => unknown) => handlers.set(event, handler),
			registerCommand: (name: string, options: { handler: (args: string, ctx: ExtensionContext) => Promise<void> | void }) => commands.set(name, options),
			registerTool: () => undefined,
		} as unknown as ExtensionAPI;
		const notifications: string[] = [];
		const ctx = {
			cwd: dataDir,
			sessionManager: {
				getSessionId: () => "session-test",
				getSessionName: () => "Test session",
			},
			ui: { notify: (message: string) => notifications.push(message) },
		} as unknown as ExtensionContext;

		try {
			await kiyosumiExtension(pi);
			await commands.get("kiyosumi-remember")!.handler("language: User prefers Japanese", ctx);
			const result = await handlers.get("before_agent_start")!({ type: "before_agent_start", prompt: "What should I remember?", systemPrompt: ["base"] } as never, ctx as never) as { systemPrompt: string[] } | undefined;
			expect(result?.systemPrompt[0]).toBe("base");
			expect(result?.systemPrompt.join("\n")).toContain("User prefers Japanese");
			expect(notifications.join("\n")).toContain("Saved language");
			await handlers.get("session_shutdown")!({ type: "session_shutdown" } as never, ctx as never);
		} finally {
			process.env.KIYOSUMI_DATA_DIR = previous.data;
			process.env.KIYOSUMI_AUTO_CONTEXT = previous.autoContext;
			if (previous.voyage === undefined) delete process.env.VOYAGE_API_KEY;
			else process.env.VOYAGE_API_KEY = previous.voyage;
			await rm(dataDir, { recursive: true, force: true });
		}
	});

	test.serial("creates a provider for a keyless custom Voyage-shaped endpoint", async () => {
		const dataDir = await mkdtemp(join(tmpdir(), "kiyosumi-extension-custom-"));
		const server = Bun.serve({ port: 0, fetch: async (request) => {
			const body = await request.json() as { input: string[] };
			return Response.json({ data: body.input.map((_text, index) => ({ index, embedding: Array.from({ length: 256 }, () => index === 0 ? 1 : 0.01) })) });
		} });
		const previous = {
			data: process.env.KIYOSUMI_DATA_DIR,
			endpoint: process.env.KIYOSUMI_EMBEDDING_ENDPOINT,
			style: process.env.KIYOSUMI_EMBEDDING_API_STYLE,
			rerank: process.env.KIYOSUMI_RERANK_ENDPOINT,
			autoContext: process.env.KIYOSUMI_AUTO_CONTEXT,
			dimensions: process.env.KIYOSUMI_EMBED_DIMENSIONS,
		};
		process.env.KIYOSUMI_DATA_DIR = dataDir;
		process.env.KIYOSUMI_EMBEDDING_ENDPOINT = `http://127.0.0.1:${server.port}/embeddings`;
		process.env.KIYOSUMI_EMBEDDING_API_STYLE = "voyage";
		process.env.KIYOSUMI_RERANK_ENDPOINT = `http://127.0.0.1:${server.port}/rerank`;
		process.env.KIYOSUMI_AUTO_CONTEXT = "0";
		process.env.KIYOSUMI_EMBED_DIMENSIONS = "256";
		delete process.env.VOYAGE_API_KEY;
		const handlers = new Map<string, (...args: never[]) => unknown>();
		const tools = new Map<string, { execute: (id: string, params: unknown, signal: AbortSignal | undefined, onUpdate: unknown, ctx: ExtensionContext) => Promise<{ content: { text: string }[] }> }>();
		const pi = {
			setLabel: () => undefined,
			zod: { object: schema, string: schema, number: schema, enum: schema },
			logger: { warn: () => undefined, debug: () => undefined },
			on: (event: string, handler: (...args: never[]) => unknown) => handlers.set(event, handler),
			registerCommand: () => undefined,
			registerTool: (tool: { name: string; execute: (id: string, params: unknown, signal: AbortSignal | undefined, onUpdate: unknown, ctx: ExtensionContext) => Promise<{ content: { text: string }[] }> }) => tools.set(tool.name, tool),
		} as unknown as ExtensionAPI;
		const ctx = { cwd: dataDir, sessionManager: { getSessionId: () => "custom-session", getSessionName: () => "Custom" }, ui: { notify: () => undefined } } as unknown as ExtensionContext;
		try {
			await kiyosumiExtension(pi);
			const result = await tools.get("kiyosumi_rag_search")!.execute("call", { query: "hello", collection: "custom" }, undefined, undefined, ctx);
			expect(result.content[0]?.text).toContain("No indexed passages");
			await handlers.get("session_shutdown")!({ type: "session_shutdown" } as never, ctx as never);
		} finally {
			server.stop(true);
			process.env.KIYOSUMI_DATA_DIR = previous.data;
			process.env.KIYOSUMI_EMBEDDING_ENDPOINT = previous.endpoint;
			process.env.KIYOSUMI_EMBEDDING_API_STYLE = previous.style;
			process.env.KIYOSUMI_RERANK_ENDPOINT = previous.rerank;
			process.env.KIYOSUMI_EMBED_DIMENSIONS = previous.dimensions;
			process.env.KIYOSUMI_AUTO_CONTEXT = previous.autoContext;
			if (previous.voyage === undefined) delete process.env.VOYAGE_API_KEY;
			else process.env.VOYAGE_API_KEY = previous.voyage;
			await rm(dataDir, { recursive: true, force: true });
		}
	});
});
