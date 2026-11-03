import { Link, Route, Routes } from "react-router-dom";
import App from "./App";
import Backdrop from "./site/Backdrop";
import Guide from "./site/Guide";
import Landing from "./site/Landing";
import NotFound from "./site/NotFound";
import Privacy from "./site/Privacy";
import SiteLayout from "./site/SiteLayout";

export default function Root() {
  return (
    <Routes>
      <Route element={<SiteLayout />}>
        <Route index element={<Landing />} />
        <Route path="guide" element={<Guide />} />
        <Route path="privacy" element={<Privacy />} />
        <Route path="*" element={<NotFound />} />
      </Route>
      <Route
        path="app/*"
        element={
          <div className="app-shell relative">
            <Backdrop />
            <header className="no-print relative z-10 border-b border-line-soft">
              <div className="mx-auto flex h-14 max-w-5xl items-center justify-between px-4 sm:px-6">
                <Link to="/" className="flex items-center gap-2.5 text-[0.8rem] font-medium uppercase tracking-[0.26em]">
                  <span className="size-2 rotate-45 bg-accent" aria-hidden />
                  PhysiqueGap
                </Link>
                <Link to="/privacy" className="text-sm text-ink-3 hover:text-ink">
                  Privacy
                </Link>
              </div>
            </header>
            <App />
          </div>
        }
      />
    </Routes>
  );
}
