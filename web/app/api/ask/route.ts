// The browser asks this page's own origin; this forwards the question to the API and streams
// the Server-Sent Events back as they arrive. API_URL is read on every request, not at build
// time, so one image works wherever the API lives.

export const dynamic = "force-dynamic";

const DEFAULT_API_URL = "http://127.0.0.1:8000";

export async function POST(request: Request): Promise<Response> {
  const base = (process.env.API_URL ?? DEFAULT_API_URL).replace(/\/+$/, "");
  let upstream: Response;
  try {
    upstream = await fetch(`${base}/api/ask`, {
      method: "POST",
      headers: {
        "Content-Type": request.headers.get("content-type") ?? "application/json",
        Accept: "text/event-stream",
      },
      body: await request.text(),
      cache: "no-store",
      // If the reader goes away, stop the upstream request too.
      signal: request.signal,
    });
  } catch {
    return Response.json({ message: `The web server couldn't reach the API at ${base}.` }, { status: 502 });
  }

  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("content-type") ?? "text/event-stream",
      // no-transform keeps compression from buffering the stream; X-Accel-Buffering turns off
      // buffering in an nginx in front, if there is one.
      "Cache-Control": "no-cache, no-transform",
      "X-Accel-Buffering": "no",
    },
  });
}
