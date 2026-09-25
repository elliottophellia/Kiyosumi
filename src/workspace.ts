import { createHash } from "node:crypto";
import { lstat, readFile, readdir, realpath, stat } from "node:fs/promises";
import { relative, resolve, sep } from "node:path";

const INDEXABLE_EXTENSIONS: Record<string, true> = {
	".c": true,
	".cc": true,
	".cpp": true,
	".cs": true,
	".css": true,
	".go": true,
	".h": true,
	".hpp": true,
	".html": true,
	".java": true,
	".js": true,
	".json": true,
	".jsx": true,
	".md": true,
	".php": true,
	".py": true,
	".rb": true,
	".rs": true,
	".sh": true,
	".sql": true,
	".toml": true,
	".ts": true,
	".tsx": true,
	".txt": true,
	".yaml": true,
	".yml": true,
};

const IGNORED_DIRECTORIES: Record<string, true> = {
	".git": true,
	".next": true,
	".venv": true,
	".omp": true,
	"__pycache__": true,
	"build": true,
	"dist": true,
	"node_modules": true,
	"target": true,
	"vendor": true,
};

export interface WorkspaceFile {
	absolutePath: string;
	relativePath: string;
	content: string;
	bytes: number;
	mtimeMs: number;
}

export interface WorkspaceIndexOptions {
	workspaceRoot: string;
	targetPath?: string;
	include?: string;
	maxFileBytes: number;
	maxFiles: number;
	maxIndexBytes: number;
}

export interface WorkspaceIndex {
	files: WorkspaceFile[];
	fingerprint: string;
	bytes: number;
}

export class WorkspaceIndexLimitError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "WorkspaceIndexLimitError";
	}
}

function insideRoot(root: string, candidate: string): boolean {
	return candidate === root || candidate.startsWith(`${root}${sep}`);
}

function extensionAllowed(filePath: string): boolean {
	const dot = filePath.lastIndexOf(".");
	return dot >= 0 && INDEXABLE_EXTENSIONS[filePath.slice(dot).toLowerCase()] === true;
}

function fileFingerprint(file: WorkspaceFile): string {
	return `${file.relativePath}\u0000${file.bytes}\u0000${file.mtimeMs}`;
}

export async function collectWorkspaceIndex(options: WorkspaceIndexOptions): Promise<WorkspaceIndex> {
	const workspaceRoot = await realpath(resolve(options.workspaceRoot));
	const requestedTarget = resolve(workspaceRoot, options.targetPath ?? ".");
	const targetRoot = await realpath(requestedTarget);
	if (!insideRoot(workspaceRoot, targetRoot)) throw new Error("index target must stay inside the workspace");

	const include = options.include?.trim() ?? "";
	const glob = include ? new Bun.Glob(include) : undefined;
	const files: WorkspaceFile[] = [];
	let bytes = 0;

	async function addFile(absolutePath: string, relativePath: string): Promise<void> {
		if (!extensionAllowed(absolutePath)) return;
		if (glob && !glob.match(relativePath.split(sep).join("/"))) return;
		const fileStat = await stat(absolutePath);
		if (fileStat.size <= 0 || fileStat.size > options.maxFileBytes) return;
		const data = await readFile(absolutePath);
		let content: string;
		try {
			content = new TextDecoder("utf-8", { fatal: true }).decode(data);
		} catch {
			return;
		}
		if (!content.trim()) return;
		if (files.length >= options.maxFiles) throw new WorkspaceIndexLimitError(`index exceeds the ${options.maxFiles} file limit`);
		bytes += fileStat.size;
		if (bytes > options.maxIndexBytes) throw new WorkspaceIndexLimitError(`index exceeds the ${options.maxIndexBytes} byte limit`);
		files.push({
			absolutePath,
			relativePath: relativePath.split(sep).join("/"),
			content,
			bytes: fileStat.size,
			mtimeMs: fileStat.mtimeMs,
		});
	}

	async function walk(directory: string): Promise<void> {
		const entries = await readdir(directory, { withFileTypes: true });
		entries.sort((left, right) => left.name.localeCompare(right.name));
		for (const entry of entries) {
			const absolutePath = resolve(directory, entry.name);
			if (entry.isDirectory()) {
				if (IGNORED_DIRECTORIES[entry.name]) continue;
				const childRealPath = await realpath(absolutePath);
				if (!insideRoot(workspaceRoot, childRealPath)) continue;
				await walk(childRealPath);
				continue;
			}
			if (!entry.isFile()) continue;
			const childRealPath = await realpath(absolutePath);
			if (!insideRoot(workspaceRoot, childRealPath)) continue;
			const relativePath = relative(targetRoot, childRealPath);
			await addFile(childRealPath, relativePath);
		}
	}

	const targetStat = await lstat(targetRoot);
	if (targetStat.isDirectory()) await walk(targetRoot);
	else await addFile(targetRoot, relative(workspaceRoot, targetRoot));
	files.sort((left, right) => left.relativePath.localeCompare(right.relativePath));
	const fingerprint = createHash("sha256");
	for (const file of files) fingerprint.update(fileFingerprint(file)).update("\n");
	return { files, fingerprint: fingerprint.digest("hex"), bytes };
}

export function isIndexablePath(filePath: string): boolean {
	return extensionAllowed(filePath);
}

export function isWithinWorkspace(workspaceRoot: string, filePath: string): boolean {
	return insideRoot(resolve(workspaceRoot), resolve(filePath));
}
