import { Ask } from "../components/Ask";

export default function Home() {
  return (
    <div className="page">
      <header className="site-header">
        <h1>grounded</h1>
        <p>
          Ask your documents. Every sentence of the answer cites the passage it came from, and a local model checks
          that the passage supports it. The demo documents are the handbook of Quillbrook Cycles, a made-up bike shop.
        </p>
      </header>
      <main>
        <Ask />
      </main>
    </div>
  );
}
