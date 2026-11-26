import { readFileSync } from "node:fs";
import { join } from "node:path";
import { vi } from "vitest";

/** A stream recorded from the real API (extractive answers, no key). */
export function fixture(name: string): string {
  return readFileSync(join(__dirname, "fixtures", name), "utf8");
}

/** One event, written the way api/grounded/app.py writes it. */
export function sse(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

/** Cut text into pieces of `size` characters, the way a network might deliver it. */
export function chunks(text: string, size: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size));
  return out;
}

/** A response body the test writes to as it goes. */
export function controlledBody() {
  const encoder = new TextEncoder();
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  const stream = new ReadableStream<Uint8Array>({
    start(c) {
      controller = c;
    },
  });
  return {
    stream,
    send(...pieces: string[]) {
      for (const piece of pieces) controller.enqueue(encoder.encode(piece));
    },
    close() {
      controller.close();
    },
  };
}

/** Replace fetch with one that answers with `body` as an event stream. */
export function mockFetch(body: ReadableStream<Uint8Array> | string, status = 200) {
  const fetchMock = vi.fn(async () => {
    const stream =
      typeof body === "string"
        ? new ReadableStream<Uint8Array>({
            start(c) {
              for (const piece of chunks(body, 61)) c.enqueue(new TextEncoder().encode(piece));
              c.close();
            },
          })
        : body;
    return new Response(stream, { status, headers: { "Content-Type": "text/event-stream" } });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}
