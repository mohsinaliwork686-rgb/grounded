import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { Ask } from "../components/Ask";
import type { RetrievalEvent } from "../lib/types";
import { chunks, controlledBody, fixture, mockFetch, sse } from "./helpers";

const BIKE = fixture("bike-return.sse");
const BIKE_RETRIEVAL = JSON.parse(BIKE.split("\n")[1].slice("data: ".length)) as RetrievalEvent;
const RIDDEN = "Can I return a bike I've already ridden?";

async function ask(question: string) {
  const user = userEvent.setup();
  render(<Ask />);
  await user.type(screen.getByLabelText("Your question"), question);
  await user.click(screen.getByRole("button", { name: "Ask" }));
  return user;
}

function sentence(text: string): HTMLElement {
  const item = screen.getByText(text).closest("li");
  if (!item) throw new Error(`no sentence item for "${text}"`);
  return item;
}

describe("a streamed answer", () => {
  it("shows the text block by block with its markers, then each sentence with its verdict", async () => {
    const body = controlledBody();
    const fetchMock = mockFetch(body.stream);
    await ask(RIDDEN);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/ask",
      expect.objectContaining({ method: "POST", body: JSON.stringify({ question: RIDDEN }) }),
    );

    // The recorded stream, cut wherever 37 characters fall, up to the first verdict.
    const firstVerdict = BIKE.indexOf("event: verdict");
    const secondText = BIKE.indexOf("event: text", BIKE.indexOf("event: text") + 1);
    body.send(...chunks(BIKE.slice(0, secondText), 37));

    const answer = await screen.findByRole("region", { name: "Answer" });
    await waitFor(() =>
      expect(answer.textContent).toContain("A bike can be returned within 30 days if it has been ridden less than 50 km."),
    );
    expect(within(answer).getAllByRole("button", { name: /^Citation/ }).map((b) => b.textContent)).toEqual(["[1]"]);
    expect(within(answer).queryByText("Supported")).toBeNull();

    // The side panel: every retrieved passage, with its score, the cited one marked.
    const panel = screen.getByRole("complementary", { name: "Retrieved passages" });
    expect(within(panel).getAllByRole("listitem")).toHaveLength(5);
    expect(within(panel).getByText("0.75")).toBeTruthy();
    expect(within(panel).getAllByText("Cited")).toHaveLength(1);

    // The second block arrives on its own, citing the second passage.
    body.send(BIKE.slice(secondText, firstVerdict));
    await waitFor(() => expect(answer.textContent).toContain("You can test ride any bike in stock."));
    expect(within(answer).getAllByRole("button", { name: /^Citation/ }).map((b) => b.textContent)).toEqual([
      "[1]",
      "[2]",
    ]);

    // Then the verdicts and done: the answer switches to one sentence per line.
    body.send(...chunks(BIKE.slice(firstVerdict), 50));
    body.close();
    await screen.findByText("Bring a photo ID, and book ahead at weekends.");

    const first = sentence("A bike can be returned within 30 days if it has been ridden less than 50 km.");
    expect(within(first).getByText("Supported")).toBeTruthy();
    expect(within(first).getByRole("button", { name: "Citation 1: Returns and refunds" })).toBeTruthy();
    const last = sentence("Bring a photo ID, and book ahead at weekends.");
    expect(within(last).getByText("Supported")).toBeTruthy();
    expect(within(last).getByRole("button", { name: "Citation 2: Visiting the shop" })).toBeTruthy();
    expect(within(answer).getAllByText("Supported")).toHaveLength(4);
    expect(answer.textContent).toContain("4 supported");
    expect(within(panel).getAllByText("Cited")).toHaveLength(2);

    // The footer line from done.
    expect(answer.textContent).toContain("Model: extractive fallback, no API key");
    expect(answer.textContent).toMatch(/verification\s[\d,]+\sms/);
    expect(screen.queryByRole("alert")).toBeNull();
  });
});

describe("a refusal", () => {
  it("shows the message and detail as an answer, not an error", async () => {
    mockFetch(fixture("price-match.sse"));
    await ask("Do you match online prices?");

    const answer = await screen.findByRole("region", { name: "Answer" });
    await within(answer).findByText("Your documents don't cover this.");
    expect(within(answer).getByText("The best passage scored 0.35, below the threshold of 0.55.")).toBeTruthy();
    expect(answer.textContent).toContain("No model called");
    expect(screen.getByRole("complementary", { name: "Retrieved passages" }).textContent).toContain(
      "Retrieval isn't confident.",
    );
    expect(screen.queryByRole("alert")).toBeNull();
  });
});

// A Claude-style answer the verifier partly rejects, scripted: the extractive fallback can't
// produce one, since it only quotes the passages.
const BIKES = BIKE_RETRIEVAL.passages[0];
const FLAGGED =
  sse("retrieval", BIKE_RETRIEVAL) +
  sse("text", { block: 0, text: "", citations: [] }) +
  sse("text", {
    block: 1,
    text: "Yes: a bike can be returned within 30 days if it has been ridden less than 50 km.",
    citations: [
      {
        passage_id: BIKES.id,
        cited_text: "A bike can be returned within 30 days if it has been ridden less than 50 km.",
      },
    ],
  }) +
  sse("text", {
    block: 2,
    text: " You get the money back in cash on the same day.",
    // The passage has a line break after "further"; this citation has a space.
    citations: [{ passage_id: BIKES.id, cited_text: "Bikes ridden further than that can't be returned" }],
  }) +
  sse("text", { block: 3, text: " Most riders find that plenty.", citations: [] }) +
  sse("verdict", {
    index: 0,
    text: "Yes: a bike can be returned within 30 days if it has been ridden less than 50 km.",
    verdict: "supported",
    entailment: 0.9612,
    checks: [{ passage_id: BIKES.id, entailment: 0.9612, neutral: 0.0311, contradiction: 0.0077 }],
  }) +
  sse("verdict", {
    index: 1,
    text: "You get the money back in cash on the same day.",
    verdict: "not_supported",
    entailment: 0.0231,
    checks: [{ passage_id: BIKES.id, entailment: 0.0231, neutral: 0.8811, contradiction: 0.0958 }],
  }) +
  sse("verdict", {
    index: 2,
    text: "Most riders find that plenty.",
    verdict: "no_citation",
    entailment: null,
    checks: [],
  }) +
  sse("done", {
    model: "claude-opus-5-5",
    timings: { retrieval_ms: 30.2, generation_ms: 2210.5, verification_ms: 412.9, total_ms: 2653.6 },
    input_tokens: 1893,
    output_tokens: 61,
  });

describe("a flagged sentence", () => {
  it("is marked not supported, with an icon and a label", async () => {
    mockFetch(FLAGGED);
    await ask(RIDDEN);

    const flagged = await waitFor(() => sentence("You get the money back in cash on the same day."));
    expect(flagged.className).toContain("sentence-not_supported");
    const mark = within(flagged).getByText("Not supported").closest(".verdict");
    expect(mark?.querySelector("svg")).not.toBeNull();
    expect(mark?.getAttribute("title")).toContain("Best entailment 0.02");
    expect(within(flagged).getByRole("button", { name: "Citation 1: Returns and refunds" })).toBeTruthy();

    expect(within(sentence("Most riders find that plenty.")).getByText("No citation")).toBeTruthy();
    expect(within(sentence("Yes: a bike can be returned within 30 days if it has been ridden less than 50 km.")).getByText("Supported")).toBeTruthy();

    const answer = screen.getByRole("region", { name: "Answer" });
    expect(answer.textContent).toContain("1 supported, 1 not supported, 1 without a citation");
    expect(answer.textContent).toContain("Model: claude-opus-5-5");
    expect(answer.textContent).toContain("1,893 tokens in, 61 out");
  });
});

describe("the citation popover", () => {
  it("shows the passage, with the cited text highlighted, when its marker has focus", async () => {
    mockFetch(FLAGGED);
    await ask(RIDDEN);

    const first = await waitFor(() =>
      sentence("Yes: a bike can be returned within 30 days if it has been ridden less than 50 km."),
    );
    const marker = within(first).getByRole("button", { name: "Citation 1: Returns and refunds" });
    expect(screen.queryByRole("tooltip")).toBeNull();

    act(() => marker.focus());
    const popover = screen.getByRole("tooltip");
    expect(marker.getAttribute("aria-describedby")).toBe(popover.id);
    expect(popover.textContent).toContain("[1] Returns and refunds");
    expect(popover.textContent).toContain("Bikes");
    expect(popover.textContent).toContain("returns.md");
    expect(popover.textContent).toContain("they are still covered by the warranty.");
    expect(popover.querySelector("mark")?.textContent).toBe(
      "A bike can be returned within 30 days if it has been ridden less than 50 km.",
    );
    expect(popover.textContent).toContain("Retrieval score 0.75");
    expect(popover.textContent).toContain("entailment for this sentence 0.96");

    // Escape closes it, and focus stays on the marker.
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("tooltip")).toBeNull();
    expect(document.activeElement).toBe(marker);
  });

  it("opens on hover too", async () => {
    mockFetch(FLAGGED);
    const user = await ask(RIDDEN);
    const flagged = await waitFor(() => sentence("You get the money back in cash on the same day."));
    await user.hover(within(flagged).getByRole("button", { name: /^Citation 1/ }));
    const popover = screen.getByRole("tooltip");
    // A cited text whose whitespace differs from the passage's is still found and highlighted.
    expect(popover.querySelector("mark")?.textContent).toBe("Bikes ridden further than that can't be returned");
  });
});

describe("errors", () => {
  it("shows a failed request plainly", async () => {
    const fetchMock = mockFetch("", 502);
    fetchMock.mockImplementation(async () =>
      Response.json({ message: "The web server couldn't reach the API at http://api:8000." }, { status: 502 }),
    );
    await ask(RIDDEN);
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("The API answered 502.");
    expect(alert.textContent).toContain("couldn't reach the API at http://api:8000");
  });

  it("shows an error event from the API", async () => {
    mockFetch(
      sse("retrieval", BIKE_RETRIEVAL) +
        sse("error", { message: "The answer model failed (APIConnectionError)." }) +
        sse("done", { model: "claude-opus-5-5", timings: { retrieval_ms: 20, total_ms: 900 }, input_tokens: null, output_tokens: null }),
    );
    await ask(RIDDEN);
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("The answer model failed (APIConnectionError).");
  });

  it("shows a network failure", async () => {
    const fetchMock = mockFetch("");
    fetchMock.mockRejectedValue(new TypeError("fetch failed"));
    await ask(RIDDEN);
    expect((await screen.findByRole("alert")).textContent).toContain("The API couldn't be reached.");
  });

  it("says so when the stream ends before done", async () => {
    mockFetch(BIKE.slice(0, BIKE.indexOf("event: verdict")));
    await ask(RIDDEN);
    expect((await screen.findByRole("alert")).textContent).toContain("The answer stopped before it finished.");
  });
});

describe("the example questions", () => {
  it("ask their question straight away", async () => {
    const fetchMock = mockFetch(fixture("price-match.sse"));
    const user = userEvent.setup();
    render(<Ask />);
    await user.click(screen.getByRole("button", { name: "Do you match online prices?" }));
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/ask",
      expect.objectContaining({ body: JSON.stringify({ question: "Do you match online prices?" }) }),
    );
    expect((screen.getByLabelText("Your question") as HTMLInputElement).value).toBe("Do you match online prices?");
    await screen.findByText("Your documents don't cover this.");
  });
});
