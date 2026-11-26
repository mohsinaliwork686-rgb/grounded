"use client";

import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { citedRanges, unwrap } from "../lib/answer";
import type { Passage } from "../lib/types";

interface Props {
  /** The passage's place in the retrieval list, 1-based, as the side panel numbers it. */
  n: number;
  passage: Passage;
  /** The text the answer cited from this passage, highlighted when it's found in it. */
  cited: string[];
  /** The verifier's entailment of this passage for the sentence, in the sentence view. */
  entailment?: number;
}

const GAP = 6;
const MARGIN = 8;

/** A citation marker, `[n]`, that shows its passage on hover and on keyboard focus. */
export function CitationMarker({ n, passage, cited, entailment }: Props) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const wrapper = useRef<HTMLSpanElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const popover = useRef<HTMLSpanElement>(null);
  const hovering = useRef(false);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const cancelClose = () => clearTimeout(closeTimer.current);
  useEffect(() => cancelClose, []);

  // Place the popover under the marker (above it if there's no room), inside the viewport.
  useLayoutEffect(() => {
    if (!open) return;
    const place = () => {
      const anchor = button.current?.getBoundingClientRect();
      const box = popover.current;
      if (!anchor || !box) return;
      const width = box.offsetWidth;
      const height = box.offsetHeight;
      const viewWidth = document.documentElement.clientWidth || window.innerWidth;
      const viewHeight = window.innerHeight;
      const left = Math.min(
        Math.max(anchor.left + anchor.width / 2 - width / 2, MARGIN),
        Math.max(viewWidth - width - MARGIN, MARGIN),
      );
      let top = anchor.bottom + GAP;
      if (top + height > viewHeight - MARGIN && anchor.top - GAP - height >= MARGIN) {
        top = anchor.top - GAP - height;
      }
      box.style.left = `${Math.round(left)}px`;
      box.style.top = `${Math.round(top)}px`;
    };
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open]);

  // Escape closes it without moving focus; so does a tap anywhere else.
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    const onPointer = (event: PointerEvent) => {
      if (!wrapper.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onPointer);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onPointer);
    };
  }, [open]);

  // A Markdown passage's heading path starts with the document's title; don't say it twice.
  const where = passage.heading_path
    .filter((heading, i) => !(i === 0 && heading === passage.title))
    .join(" › ");
  const ranges = citedRanges(passage.text, cited);

  return (
    <span
      className="citation"
      ref={wrapper}
      onMouseEnter={() => {
        hovering.current = true;
        cancelClose();
        setOpen(true);
      }}
      onMouseLeave={() => {
        hovering.current = false;
        cancelClose();
        // A short delay, so the pointer can cross the gap into the popover.
        closeTimer.current = setTimeout(() => {
          if (document.activeElement !== button.current) setOpen(false);
        }, 150);
      }}
    >
      <button
        type="button"
        ref={button}
        className="citation-marker"
        aria-label={`Citation ${n}: ${passage.title}`}
        aria-describedby={open ? id : undefined}
        onFocus={() => setOpen(true)}
        onBlur={() => {
          if (!hovering.current) setOpen(false);
        }}
        onClick={() => setOpen(true)}
      >
        [{n}]
      </button>
      {open && (
        <span className="popover" role="tooltip" id={id} ref={popover}>
          <span className="popover-source">
            <span className="popover-title">
              [{n}] {passage.title}
            </span>
            <span className="popover-where">
              {[where, passage.page !== null ? `page ${passage.page}` : "", passage.document]
                .filter(Boolean)
                .join(" · ")}
            </span>
          </span>
          <span className="popover-text">
            <Highlighted text={unwrap(passage.text)} ranges={ranges} />
          </span>
          <span className="popover-meta">
            Retrieval score {passage.score.toFixed(2)}
            {entailment !== undefined && <> · entailment for this sentence {entailment.toFixed(2)}</>}
            {ranges.length > 0 && <> · cited text highlighted</>}
          </span>
        </span>
      )}
    </span>
  );
}

function Highlighted({ text, ranges }: { text: string; ranges: [number, number][] }) {
  const parts = [];
  let cursor = 0;
  for (const [start, end] of ranges) {
    if (start > cursor) parts.push(text.slice(cursor, start));
    parts.push(<mark key={start}>{text.slice(start, end)}</mark>);
    cursor = end;
  }
  if (cursor < text.length) parts.push(text.slice(cursor));
  return <>{parts}</>;
}
