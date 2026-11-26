// A small Server-Sent Events parser for a fetch() body. EventSource can only GET, and
// /api/ask is a POST, so the page reads the stream itself.
//
// It follows the parts of the SSE format the API uses: `event:` and `data:` fields, events
// separated by a blank line, any line ending (\n, \r\n or \r), comments starting with ":".
// Chunks can end anywhere, even between the \r and \n of one line ending.

export interface SseMessage {
  event: string;
  data: string;
}

export class SseParser {
  private buffer = "";
  private eventName = "";
  private data: string[] = [];

  /** Feed the next piece of text; returns the events it completed, in order. */
  push(chunk: string): SseMessage[] {
    this.buffer += chunk;
    const messages: SseMessage[] = [];
    let start = 0;
    for (;;) {
      const lf = this.buffer.indexOf("\n", start);
      const cr = this.buffer.indexOf("\r", start);
      let end: number;
      let next: number;
      if (cr !== -1 && (lf === -1 || cr < lf)) {
        // A \r at the very end may be the first half of \r\n: wait for the next chunk.
        if (cr === this.buffer.length - 1) break;
        end = cr;
        next = this.buffer[cr + 1] === "\n" ? cr + 2 : cr + 1;
      } else if (lf !== -1) {
        end = lf;
        next = lf + 1;
      } else {
        break;
      }
      const message = this.line(this.buffer.slice(start, end));
      if (message) messages.push(message);
      start = next;
    }
    this.buffer = this.buffer.slice(start);
    return messages;
  }

  /** At the end of the stream: a \r held back as the possible start of \r\n was a line end. */
  end(): SseMessage[] {
    return this.buffer.endsWith("\r") ? this.push("\n") : [];
  }

  private line(line: string): SseMessage | null {
    if (line === "") {
      const message =
        this.data.length > 0 ? { event: this.eventName || "message", data: this.data.join("\n") } : null;
      this.eventName = "";
      this.data = [];
      return message;
    }
    if (line.startsWith(":")) return null;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") this.eventName = value;
    else if (field === "data") this.data.push(value);
    return null;
  }
}

/** The events of a byte stream, decoded as UTF-8 (a character split across chunks is fine). */
export async function* readSse(body: ReadableStream<Uint8Array>): AsyncGenerator<SseMessage> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const parser = new SseParser();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      yield* parser.push(decoder.decode(value, { stream: true }));
    }
    yield* parser.push(decoder.decode());
    // An event the stream ended in the middle of is dropped, as the SSE format says.
    yield* parser.end();
  } finally {
    // Closes the connection if the reader stopped early; harmless once the stream has ended.
    reader.cancel().catch(() => undefined);
  }
}
