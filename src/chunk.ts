import { createHash } from "node:crypto";

export function chunkText(content: string, size = 1200, overlap = 200): string[] {
	const normalized = content.replaceAll("\r\n", "\n");
	const windowSize = size > 0 ? size : 1200;
	const windowOverlap = overlap >= 0 && overlap < windowSize ? overlap : Math.floor(windowSize / 6);
	const characters = Array.from(normalized);
	if (characters.length === 0) return [];
	if (characters.length <= windowSize) {
		const only = normalized.trim();
		return only ? [only] : [];
	}

	const chunks: string[] = [];
	let start = 0;
	while (start < characters.length) {
		const desiredEnd = Math.min(start + windowSize, characters.length);
		if (desiredEnd === characters.length) {
			const tail = characters.slice(start).join("").trim();
			if (tail) chunks.push(tail);
			break;
		}

		const floor = start + Math.floor(windowSize * 2 / 3);
		let cut = desiredEnd;
		for (let index = desiredEnd; index > floor; index--) {
			if (characters[index] === "\n" && characters[index - 1] === "\n") {
				cut = index;
				break;
			}
		}
		if (cut === desiredEnd) {
			for (let index = desiredEnd; index > floor; index--) {
				if (characters[index] === "\n") {
					cut = index;
					break;
				}
			}
		}

		const piece = characters.slice(start, cut).join("").trim();
		if (piece) chunks.push(piece);
		const next = cut - windowOverlap;
		start = next > start ? next : cut;
	}
	return chunks;
}

export function chunkId(collection: string, documentId: string, index: number): string {
	const digest = createHash("sha256").update(`${collection}|${documentId}|${index}`).digest("hex").slice(0, 24);
	return `chk_${digest}`;
}
