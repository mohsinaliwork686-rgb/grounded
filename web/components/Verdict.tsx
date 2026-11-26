import type { Verdict } from "../lib/types";
import { NoCitationIcon, NotSupportedIcon, SupportedIcon } from "./icons";

const MARKS = {
  supported: { label: "Supported", Icon: SupportedIcon },
  not_supported: { label: "Not supported", Icon: NotSupportedIcon },
  no_citation: { label: "No citation", Icon: NoCitationIcon },
} as const;

export function verdictLabel(verdict: Verdict): string | null {
  return verdict === "connective" ? null : MARKS[verdict].label;
}

function explain(verdict: Verdict, entailment: number | null): string {
  const score = entailment === null ? null : entailment.toFixed(2);
  switch (verdict) {
    case "supported":
      return `Entailment ${score}: a cited passage supports this sentence.`;
    case "not_supported":
      return score === null
        ? "No cited passage supports this sentence."
        : `Best entailment ${score}: no cited passage supports this sentence.`;
    case "no_citation":
      return "This sentence cites no passage, so there is nothing to check it against.";
    default:
      return "";
  }
}

/** The verifier's mark for one sentence: an icon and a label, never colour alone. */
export function VerdictMark({ verdict, entailment }: { verdict: Verdict; entailment: number | null }) {
  if (verdict === "connective") return null;
  const { label, Icon } = MARKS[verdict];
  const detail = explain(verdict, entailment);
  return (
    <span className={`verdict verdict-${verdict}`} title={detail}>
      <Icon />
      <span className="verdict-label">{label}</span>
      <span className="visually-hidden">. {detail}</span>
    </span>
  );
}
