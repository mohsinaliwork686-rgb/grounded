import { unwrap } from "../lib/answer";
import type { RetrievalEvent } from "../lib/types";
import { CitedIcon } from "./icons";

function clamp(score: number): number {
  return Math.min(Math.max(score, 0), 1);
}

/** Every retrieved passage with its score, so you can see what the answer was working from. */
export function Passages({ retrieval, cited }: { retrieval: RetrievalEvent; cited: Set<number> }) {
  const { passages, top_score: top, threshold, confident } = retrieval;
  return (
    <aside className="panel" aria-labelledby="panel-heading">
      <h2 id="panel-heading">Retrieved passages</h2>
      {confident ? (
        <p className="panel-note">
          The best passage scored {top.toFixed(2)}, above the refusal threshold of {threshold.toFixed(2)}.
        </p>
      ) : (
        <p className="panel-note panel-note-weak">
          <strong>Retrieval isn&apos;t confident.</strong> The best passage scored {top.toFixed(2)}, below the
          threshold of {threshold.toFixed(2)}.
        </p>
      )}
      <p className="panel-hint">
        Each bar is a passage&apos;s similarity to the question, from 0 to 1. The line marks the threshold.
      </p>
      <ol className="passage-list">
        {passages.map((passage, i) => {
          const isCited = cited.has(passage.id);
          const where = passage.heading_path
            .filter((heading, at) => !(at === 0 && heading === passage.title))
            .join(" › ");
          const above = passage.score >= threshold;
          return (
            <li key={passage.id} className={isCited ? "passage passage-cited" : "passage"}>
              <div className="passage-head">
                <span className="passage-title">
                  <span className="passage-number">[{i + 1}]</span> {passage.title}
                </span>
                {isCited && (
                  <span className="cited-badge">
                    <CitedIcon />
                    Cited
                  </span>
                )}
              </div>
              <p className="passage-where">
                {[where, passage.page !== null ? `page ${passage.page}` : "", passage.document]
                  .filter(Boolean)
                  .join(" · ")}
              </p>
              <div className="score">
                <span className="score-bar" aria-hidden="true">
                  <span
                    className={above ? "score-fill" : "score-fill score-fill-below"}
                    style={{ width: `${clamp(passage.score) * 100}%` }}
                  />
                  <span className="score-threshold" style={{ left: `${clamp(threshold) * 100}%` }} />
                </span>
                <span className="score-value">
                  <span className="visually-hidden">Score </span>
                  {passage.score.toFixed(2)}
                  <span className="visually-hidden">{above ? ", above the threshold" : ", below the threshold"}</span>
                </span>
              </div>
              <p className="passage-text">{unwrap(passage.text)}</p>
            </li>
          );
        })}
      </ol>
    </aside>
  );
}
