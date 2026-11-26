import { describe, expect, it } from "vitest";
import { readSse, SseParser, type SseMessage } from "../lib/sse";
import { fixture } from "./helpers";

function parseAll(pieces: string[]): SseMessage[] {
  const parser = new SseParser();
  return [...pieces.flatMap((piece) => parser.push(piece)), ...parser.end()];
}

const TWO_EVENTS =
  'event: text\ndata: {"block": 0, "text": "Hello", "citations": []}\n\n' +
  'event: done\ndata: {"model": "extractive"}\n\n';

const EXPECTED = [
  { event: "text", data: '{"block": 0, "text": "Hello", "citations": []}' },
  { event: "done", data: '{"model": "extractive"}' },
];

describe("SseParser", () => {
  it("reads several events from one chunk", () => {
    expect(parseAll([TWO_EVENTS])).toEqual(EXPECTED);
  });

  it("reads an event split across chunks", () => {
    const pieces = ["event: te", 'xt\ndata: {"block": 0, "te', 'xt": "Hello", "citations": []}\n', "\n"];
    const parser = new SseParser();
    expect(parser.push(pieces[0])).toEqual([]);
    expect(parser.push(pieces[1])).toEqual([]);
    expect(parser.push(pieces[2])).toEqual([]);
    expect(parser.push(pieces[3])).toEqual([EXPECTED[0]]);
  });

  it("gives the same events wherever the stream is cut", () => {
    for (let cut = 0; cut <= TWO_EVENTS.length; cut++) {
      expect(parseAll([TWO_EVENTS.slice(0, cut), TWO_EVENTS.slice(cut)])).toEqual(EXPECTED);
    }
    expect(parseAll([...TWO_EVENTS])).toEqual(EXPECTED);
  });

  it("handles \\r\\n line endings, even split between the \\r and the \\n", () => {
    const crlf = TWO_EVENTS.replaceAll("\n", "\r\n");
    const at = crlf.indexOf("\r\n\r\n") + 1;
    expect(parseAll([crlf.slice(0, at), crlf.slice(at)])).toEqual(EXPECTED);
    expect(parseAll([TWO_EVENTS.replaceAll("\n", "\r")])).toEqual(EXPECTED);
  });

  it("joins multi-line data, ignores comments and names unnamed events 'message'", () => {
    const text = ": keep-alive\n\ndata: first\ndata: second\n\nevent: x\ndata:no space\n\n";
    expect(parseAll([text])).toEqual([
      { event: "message", data: "first\nsecond" },
      { event: "x", data: "no space" },
    ]);
  });

  it("holds back an event until its blank line arrives", () => {
    const parser = new SseParser();
    expect(parser.push('event: done\ndata: {"model": null}\n')).toEqual([]);
    expect(parser.push("\n")).toEqual([{ event: "done", data: '{"model": null}' }]);
  });
});

describe("readSse", () => {
  it("decodes a character split across byte chunks", async () => {
    const bytes = new TextEncoder().encode(fixture("price-match.sse"));
    const pound = bytes.indexOf(0xc2); // the first byte of "£"
    expect(pound).toBeGreaterThan(0);
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(bytes.slice(0, pound + 1));
        controller.enqueue(bytes.slice(pound + 1));
        controller.close();
      },
    });
    const events: SseMessage[] = [];
    for await (const event of readSse(body)) events.push(event);
    expect(events.map((e) => e.event)).toEqual(["retrieval", "refused", "done"]);
    expect(JSON.parse(events[0].data).passages[0].text).toContain("£20");
  });
});
