import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <main className="mx-auto max-w-xl px-4 py-32 text-center">
      <p className="text-sm text-ink-3">404</p>
      <h1 className="mt-2 text-4xl font-medium tracking-tight">That page is not here</h1>
      <Link to="/" className="btn btn-ghost mt-8">
        Back to the start
      </Link>
    </main>
  );
}
