import { describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, rename, rm } from "node:fs/promises";
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
			await commands.get("kiyosumi")!.handler("memory save language: User prefers Japanese", ctx);
			const result = await handlers.get("before_agent_start")!({ type: "before_agent_start", prompt: "What should I remember?", systemPrompt: ["base"] } as never, ctx as never) as { systemPrompt: string[] } | undefined;
			expect(result?.systemPrompt[0]).toBe("base");
			expect(result?.systemPrompt.join("\n")).toContain("User prefers Japanese");
			expect([...commands.keys()]).toEqual(["kiyosumi"]);
			await handlers.get("session_shutdown")!({ type: "session_shutdown" } as never, ctx as never);
		} finally {
			process.env.KIYOSUMI_DATA_DIR = previous.data ?? "";
			if (previous.autoContext === undefined) delete process.env.KIYOSUMI_AUTO_CONTEXT;
			else process.env.KIYOSUMI_AUTO_CONTEXT = previous.autoContext;
			if (previous.voyage === undefined) delete process.env.VOYAGE_API_KEY;
			else process.env.VOYAGE_API_KEY = previous.voyage;
			await rm(dataDir, { recursive: true, force: true });
		}
	});
	test.serial("keeps an active runtime usable when its peer shuts down during aborted-turn indexing", async () => {
		const dataDir = await mkdtemp(join(tmpdir(), "kiyosumi-extension-abort-overlap-"));
		const previous = {
			data: process.env.KIYOSUMI_DATA_DIR,
			endpoint: process.env.KIYOSUMI_EMBEDDING_ENDPOINT,
			style: process.env.KIYOSUMI_EMBEDDING_API_STYLE,
			rerank: process.env.KIYOSUMI_RERANK_ENDPOINT,
			dimensions: process.env.KIYOSUMI_EMBED_DIMENSIONS,
			autoContext: process.env.KIYOSUMI_AUTO_CONTEXT,
			voyage: process.env.VOYAGE_API_KEY,
		};
		const { promise: blocked, resolve: release } = Promise.withResolvers<void>();
		const { promise: bothStarted, resolve: resolveBothStarted } = Promise.withResolvers<void>();
		let embeddingsStarted = 0;
		const warnings: Array<{ task: string; error: string; stack?: string }> = [];
		const server = Bun.serve({ port: 0, fetch: async (request) => {
			if (new URL(request.url).pathname.endsWith("/rerank")) return Response.json({ data: [] });
			const body = await request.json() as { input: string[] };
			embeddingsStarted += 1;
			if (embeddingsStarted === 2) resolveBothStarted();
			await blocked;
			return Response.json({ data: body.input.map((_text, index) => ({ index, embedding: Array.from({ length: 256 }, (_unused, dimension) => dimension === 0 ? 1 : 0.01) })) });
		} });
		process.env.KIYOSUMI_DATA_DIR = dataDir;
		process.env.KIYOSUMI_EMBEDDING_ENDPOINT = `http://127.0.0.1:${server.port}/embeddings`;
		process.env.KIYOSUMI_EMBEDDING_API_STYLE = "voyage";
		process.env.KIYOSUMI_RERANK_ENDPOINT = `http://127.0.0.1:${server.port}/rerank`;
		process.env.KIYOSUMI_EMBED_DIMENSIONS = "256";
		process.env.KIYOSUMI_AUTO_CONTEXT = "1";
		delete process.env.VOYAGE_API_KEY;
		const makeRuntime = () => {
			const handlers = new Map<string, (...args: never[]) => unknown>();
			const pi = {
				setLabel: () => undefined,
				zod: { object: schema, string: schema, number: schema, enum: schema },
				logger: { warn: (_message: string, details: { task: string; error: string; stack?: string }) => warnings.push(details), debug: () => undefined },
				on: (event: string, handler: (...args: never[]) => unknown) => handlers.set(event, handler),
				registerCommand: () => undefined,
				registerTool: () => undefined,
			} as unknown as ExtensionAPI;
			return { pi, handlers };
		};
		const first = makeRuntime();
		const second = makeRuntime();
		const context = (sessionId: string) => ({ cwd: dataDir, sessionManager: { getSessionId: () => sessionId, getSessionName: () => sessionId }, ui: { notify: () => undefined }, setTimeout, clearTimeout }) as unknown as ExtensionContext;
		const firstContext = context("first-session");
		const secondContext = context("second-session");
		try {
			await Promise.all([kiyosumiExtension(first.pi), kiyosumiExtension(second.pi)]);
			first.handlers.get("agent_end")!({ type: "agent_end", willContinue: false, messages: [{ role: "user", content: "first aborted turn" }, { role: "assistant", content: "", stopReason: "aborted" }] } as never, firstContext as never);
			second.handlers.get("agent_end")!({ type: "agent_end", willContinue: false, messages: [{ role: "user", content: "second aborted turn" }, { role: "assistant", content: "", stopReason: "aborted" }] } as never, secondContext as never);
			await bothStarted;
			expect(embeddingsStarted).toBe(2);
			const firstShutdown = first.handlers.get("session_shutdown")!({ type: "session_shutdown" } as never, firstContext as never) as Promise<void>;
			release();
			await firstShutdown;
			await second.handlers.get("before_agent_start")!({ type: "before_agent_start", prompt: "still usable after peer shutdown", systemPrompt: ["base"] } as never, secondContext as never);
			await second.handlers.get("session_shutdown")!({ type: "session_shutdown" } as never, secondContext as never);
			expect(warnings).toEqual([]);
		} finally {
			release();
			server.stop(true);
			if (previous.data === undefined) delete process.env.KIYOSUMI_DATA_DIR;
			else process.env.KIYOSUMI_DATA_DIR = previous.data;
			if (previous.endpoint === undefined) delete process.env.KIYOSUMI_EMBEDDING_ENDPOINT;
			else process.env.KIYOSUMI_EMBEDDING_ENDPOINT = previous.endpoint;
			if (previous.style === undefined) delete process.env.KIYOSUMI_EMBEDDING_API_STYLE;
			else process.env.KIYOSUMI_EMBEDDING_API_STYLE = previous.style;
			if (previous.rerank === undefined) delete process.env.KIYOSUMI_RERANK_ENDPOINT;
			else process.env.KIYOSUMI_RERANK_ENDPOINT = previous.rerank;
			if (previous.dimensions === undefined) delete process.env.KIYOSUMI_EMBED_DIMENSIONS;
			else process.env.KIYOSUMI_EMBED_DIMENSIONS = previous.dimensions;
			if (previous.autoContext === undefined) delete process.env.KIYOSUMI_AUTO_CONTEXT;
			else process.env.KIYOSUMI_AUTO_CONTEXT = previous.autoContext;
			if (previous.voyage === undefined) delete process.env.VOYAGE_API_KEY;
			else process.env.VOYAGE_API_KEY = previous.voyage;
			await rm(dataDir, { recursive: true, force: true });
		}
	});

	test.serial("routes consolidated workspace and memory commands", async () => {
		const dataDir = await mkdtemp(join(tmpdir(), "kiyosumi-command-data-"));
		const workspace = await mkdtemp(join(tmpdir(), "kiyosumi-command-project-"));
		await Bun.write(join(workspace, "README.md"), "Synthetic command fixture overview sentence.");
		await Bun.write(join(workspace, "notes.txt"), "Unique searchable project passage.");
		await Bun.write(join(workspace, "obsolete.txt"), "Obsolete project passage for replacement check.");
		await Bun.write(join(workspace, "other.txt"), "Unrelated preserved passage.");
		await mkdir(join(workspace, "docs with space"));
		const previous = { data: process.env.KIYOSUMI_DATA_DIR, endpoint: process.env.KIYOSUMI_EMBEDDING_ENDPOINT, style: process.env.KIYOSUMI_EMBEDDING_API_STYLE, rerank: process.env.KIYOSUMI_RERANK_ENDPOINT, rerankEnabled: process.env.KIYOSUMI_RERANK, ragEnabled: process.env.KIYOSUMI_RAG_ENABLED, dimensions: process.env.KIYOSUMI_EMBED_DIMENSIONS, autoContext: process.env.KIYOSUMI_AUTO_CONTEXT, voyage: process.env.VOYAGE_API_KEY };
		await Bun.write(join(workspace, "docs with space", "guide.md"), "A spaced directory contains this guide.");
		const server = Bun.serve({ port: 0, fetch: async (request) => {
			if (new URL(request.url).pathname.endsWith("/rerank")) return Response.json({ data: [] });
			const body = await request.json() as { input: string[] };
			return Response.json({ data: body.input.map((_text, index) => ({ index, embedding: Array.from({ length: 256 }, (_unused, dimension) => dimension === 0 ? 1 : 0.01) })) });
		} });
		process.env.KIYOSUMI_DATA_DIR = dataDir;
		process.env.KIYOSUMI_EMBEDDING_ENDPOINT = `http://127.0.0.1:${server.port}/embeddings`;
		process.env.KIYOSUMI_EMBEDDING_API_STYLE = "voyage";
		process.env.KIYOSUMI_RERANK_ENDPOINT = `http://127.0.0.1:${server.port}/rerank`;
		process.env.KIYOSUMI_EMBED_DIMENSIONS = "256";
		process.env.KIYOSUMI_RERANK = "0";
		delete process.env.VOYAGE_API_KEY;
		const handlers = new Map<string, (...args: never[]) => unknown>();
		const commands = new Map<string, { handler: (args: string, ctx: ExtensionContext) => Promise<void> | void }>();
		const notifications: string[] = [];
		const messages: string[] = [];
		const pi = { setLabel: () => undefined, zod: { object: schema, string: schema, number: schema, enum: schema }, logger: { warn: () => undefined, debug: () => undefined }, on: (event: string, handler: (...args: never[]) => unknown) => handlers.set(event, handler), registerCommand: (name: string, options: { handler: (args: string, ctx: ExtensionContext) => Promise<void> | void }) => commands.set(name, options), registerTool: () => undefined, sendUserMessage: (message: string) => messages.push(message) } as unknown as ExtensionAPI;
		const ctx = { cwd: workspace, sessionManager: { getSessionId: () => "command-session", getSessionName: () => "Command" }, ui: { notify: (message: string) => notifications.push(message) } } as unknown as ExtensionContext;
		const assertOutput = (expected: string) => {
			const actual = notifications.at(-1);
			if (!actual?.includes(expected)) throw new Error(`Expected command output to include ${expected}; got ${actual ?? "<none>"}`);
		};
		try {
			await kiyosumiExtension(pi);
			const command = commands.get("kiyosumi")!.handler;
			await command("help", ctx);
			assertOutput("/kiyosumi analyze");
			await command("memory save language: Japanese preferred", ctx);
			await command("memory search Japanese preferred", ctx);
			assertOutput("Japanese preferred");
			await command("memory list", ctx);
			assertOutput("language");
			expect(handlers.has("session_start")).toBe(false);
			await command("index", ctx);
			await command("search Unrelated preserved passage", ctx);
			assertOutput("Unrelated preserved passage");
			await rm(join(workspace, "obsolete.txt"));
			await command("index", ctx);
			await command("search Obsolete project passage for replacement check", ctx);
			if (notifications.at(-1)?.includes("Obsolete project passage")) throw new Error("Full-workspace reindex retained a removed document.");
			await command("index docs with space", ctx);
			await command("search spaced directory contains this guide", ctx);
			assertOutput("spaced directory contains this guide");
			await command("search Unrelated preserved passage", ctx);
			assertOutput("Unrelated preserved passage");
			await command("memory delete language", ctx);
			assertOutput("Forgot language");
			await command("delete-index", ctx);
			await command("search absent phrase", ctx);
			assertOutput("Run /kiyosumi analyze or /kiyosumi index first");
			const movedWorkspace = `${workspace}-moved`;
			await rename(workspace, movedWorkspace);
			await command("analyze", ctx);
			expect(String(messages.at(-1)).includes("read-only")).toBe(true);
			await rename(movedWorkspace, workspace);
			await handlers.get("session_shutdown")!({ type: "session_shutdown" } as never, ctx as never);
		} finally {
			server.stop(true);
			for (const [key, value] of Object.entries({ KIYOSUMI_DATA_DIR: previous.data, KIYOSUMI_EMBEDDING_ENDPOINT: previous.endpoint, KIYOSUMI_EMBEDDING_API_STYLE: previous.style, KIYOSUMI_RERANK_ENDPOINT: previous.rerank, KIYOSUMI_RERANK: previous.rerankEnabled, KIYOSUMI_RAG_ENABLED: previous.ragEnabled, KIYOSUMI_EMBED_DIMENSIONS: previous.dimensions, KIYOSUMI_AUTO_CONTEXT: previous.autoContext, VOYAGE_API_KEY: previous.voyage })) {
				if (value === undefined) delete process.env[key];
				else process.env[key] = value;
			}
			await rm(dataDir, { recursive: true, force: true });
			await rm(workspace, { recursive: true, force: true });
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
			for (const [key, value] of Object.entries({ KIYOSUMI_DATA_DIR: previous.data, KIYOSUMI_EMBEDDING_ENDPOINT: previous.endpoint, KIYOSUMI_EMBEDDING_API_STYLE: previous.style, KIYOSUMI_RERANK_ENDPOINT: previous.rerank, KIYOSUMI_EMBED_DIMENSIONS: previous.dimensions, KIYOSUMI_AUTO_CONTEXT: previous.autoContext, VOYAGE_API_KEY: previous.voyage })) {
				if (value === undefined) delete process.env[key];
				else process.env[key] = value;
			}
			await rm(dataDir, { recursive: true, force: true });
		}
	});
});
