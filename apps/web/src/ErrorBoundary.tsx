import { Component, type ErrorInfo, type ReactNode } from "react";

/** Last line of defence: a render error shows a message instead of a blank page. Details only in dev. */
export default class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <main style={{ maxWidth: 640, margin: "0 auto", padding: "6rem 1.5rem", color: "#f4efe6" }}>
        <h1 style={{ fontSize: "1.75rem", fontWeight: 500 }}>Something went wrong</h1>
        <p style={{ marginTop: "1rem", color: "#cfc7b9" }}>Reload the page. If it keeps happening, the app has a bug.</p>
        {import.meta.env.DEV && (
          <pre style={{ marginTop: "1.5rem", whiteSpace: "pre-wrap", color: "#e8d3a6", fontSize: 13 }}>
            {error.message}
            {"\n\n"}
            {error.stack}
          </pre>
        )}
      </main>
    );
  }
}
