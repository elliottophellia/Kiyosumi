import { describe, expect, test } from "bun:test";
import { chunkId, chunkText } from "../src/chunk.ts";

 describe("chunkText", () => {
	test("normalizes line endings and keeps short text intact", () => {
		expect(chunkText("  hello\r\nworld  ", 100, 10)).toEqual(["hello\nworld"]);
	});

	test("respects Unicode code points and overlap", () => {
		const content = "😀".repeat(450) + "\n\n" + "終".repeat(450);
		const chunks = chunkText(content, 200, 40);
		expect(chunks.length).toBeGreaterThan(2);
		expect(chunks.every((chunk) => Array.from(chunk).length <= 200)).toBe(true);
		expect(chunks.join("")).toContain("終");
	});

	test("does not loop when overlap is close to the window size", () => {
		const chunks = chunkText("a".repeat(1_000), 200, 199);
		expect(chunks.length).toBeGreaterThan(1);
		expect(chunks.every((chunk) => chunk.length > 0)).toBe(true);
	});

	test("creates stable, collection-scoped chunk identifiers", () => {
		expect(chunkId("notes", "a.md", 0)).toBe(chunkId("notes", "a.md", 0));
		expect(chunkId("notes", "a.md", 0)).not.toBe(chunkId("other", "a.md", 0));
	});
});
