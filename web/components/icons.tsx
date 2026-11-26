// Small inline icons, so the page needs no icon font or network request. Each is decorative:
// the text next to it says the same thing.

import type { ReactNode } from "react";

function Icon({ children }: { children: ReactNode }) {
  return (
    <svg
      className="icon"
      viewBox="0 0 16 16"
      width="16"
      height="16"
      aria-hidden="true"
      focusable="false"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {children}
    </svg>
  );
}

export function SupportedIcon() {
  return (
    <Icon>
      <circle cx="8" cy="8" r="6.5" />
      <path d="M5 8.2l2 2 4-4.4" />
    </Icon>
  );
}

export function NotSupportedIcon() {
  return (
    <Icon>
      <path d="M8 1.8l6.4 11.4H1.6z" />
      <path d="M8 6.2v3.3" />
      <path d="M8 11.6v.1" />
    </Icon>
  );
}

export function NoCitationIcon() {
  return (
    <Icon>
      <circle cx="8" cy="8" r="6.5" />
      <path d="M5 8h6" />
    </Icon>
  );
}

export function InfoIcon() {
  return (
    <Icon>
      <circle cx="8" cy="8" r="6.5" />
      <path d="M8 7.2v4" />
      <path d="M8 4.8v.1" />
    </Icon>
  );
}

export function ErrorIcon() {
  return (
    <Icon>
      <circle cx="8" cy="8" r="6.5" />
      <path d="M5.8 5.8l4.4 4.4M10.2 5.8l-4.4 4.4" />
    </Icon>
  );
}

export function CitedIcon() {
  return (
    <Icon>
      <path d="M3.5 8.4l3 3 6-6.8" />
    </Icon>
  );
}
