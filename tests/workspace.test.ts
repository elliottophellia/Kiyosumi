import { describe, expect, test } from "bun:test";
import { mkdtemp, mkdir, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { collectWorkspaceIndex, isWithinWorkspace, WorkspaceIndexLimitError } from "../src/workspace.ts";

async function fixture(): Promise<string> {
	const root = await mkdtemp(join(tmpdir(), "kiyosumi-workspace-"));
	await mkdir(join(root, "src"));
	await mkdir(join(root, "node_modules", "ignored"), { recursive: true });
	await writeFile(join(root, "README.md"), "# Kiyosumi\n\nA memory plugin for Oh My Pi.\n");
	await writeFile(join(root, "src", "main.ts"), "export const answer = 42;\n");
	await writeFile(join(root, "node_modules", "ignored", "bad.txt"), "do not index\n");
	return root;
}

const limits = { maxFileBytes: 10_000, maxFiles: 20, maxIndexBytes: 100_000 };

describe("workspace collection", () => {
	test("walks supported text files and ignores dependency directories", async () => {
		const root = await fixture();
		const index = await collectWorkspaceIndex({ workspaceRoot: root, ...limits });
		expect(index.files.map((file) => file.relativePath)).toEqual(["README.md", "src/main.ts"]);
		expect(index.fingerprint).toHaveLength(64);
	});

	test("applies include patterns relative to the target directory", async () => {
		const root = await fixture();
		const index = await collectWorkspaceIndex({ workspaceRoot: root, targetPath: "src", include: "**/*.ts", ...limits });
		expect(index.files.map((file) => file.relativePath)).toEqual(["main.ts"]);
	});

	test("keeps the workspace-relative path for a single file target", async () => {
		const root = await fixture();
		const index = await collectWorkspaceIndex({ workspaceRoot: root, targetPath: "src/main.ts", ...limits });
		expect(index.files.map((file) => file.relativePath)).toEqual(["src/main.ts"]);
	});

	test("rejects targets that resolve outside the workspace", async () => {
		const root = await fixture();
		const outside = await mkdtemp(join(tmpdir(), "kiyosumi-outside-"));
		await symlink(outside, join(root, "escape"));
		await expect(collectWorkspaceIndex({ workspaceRoot: root, targetPath: "escape", ...limits })).rejects.toThrow("inside the workspace");
	});

	test("enforces file and byte limits before embedding and persistence", async () => {
		const root = await fixture();
		await expect(collectWorkspaceIndex({ workspaceRoot: root, maxFileBytes: 10_000, maxFiles: 1, maxIndexBytes: 100_000 })).rejects.toBeInstanceOf(WorkspaceIndexLimitError);
	});

	test("does not charge skipped binary files against limits", async () => {
		const root = await fixture();
		await writeFile(join(root, "binary.md"), new Uint8Array([0xff, 0xfe, 0xfd]));
		const index = await collectWorkspaceIndex({ workspaceRoot: root, maxFileBytes: 10_000, maxFiles: 2, maxIndexBytes: 100_000 });
		expect(index.files.map((file) => file.relativePath)).toEqual(["README.md", "src/main.ts"]);
	});

	test("recognizes only paths inside the workspace", () => {
		expect(isWithinWorkspace("/tmp/project", "/tmp/project/src/file.ts")).toBe(true);
		expect(isWithinWorkspace("/tmp/project", "/tmp/project-other/file.ts")).toBe(false);
	});
});
