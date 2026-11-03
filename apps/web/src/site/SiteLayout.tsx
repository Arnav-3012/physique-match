import { useEffect } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import Backdrop from "./Backdrop";

const link = ({ isActive }: { isActive: boolean }) =>
  `text-sm font-medium transition-colors ${isActive ? "text-ink" : "text-ink-3 hover:text-ink"}`;

export default function SiteLayout() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);

  return (
    <div className="relative min-h-screen bg-canvas text-ink">
      <Backdrop />
      <header className="sticky top-0 z-40 border-b border-line-soft bg-canvas/70 backdrop-blur-md">
        <nav className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4 sm:px-6" aria-label="Main">
          <Link to="/" className="flex items-center gap-2.5 text-[0.8rem] font-medium uppercase tracking-[0.26em]">
            <span className="size-2 rotate-45 bg-accent" aria-hidden />
            PhysiqueGap
          </Link>
          <div className="flex items-center gap-5">
            <NavLink to="/guide" className={link}>
              Guide
            </NavLink>
            <NavLink to="/privacy" className={link}>
              Privacy
            </NavLink>
            <Link to="/app" className="btn btn-primary !px-3.5 !py-2 !text-sm">
              Start scan
            </Link>
          </div>
        </nav>
      </header>

      <div className="relative z-10">
        <Outlet />
      </div>

      <footer className="relative z-10 border-t border-line-soft bg-canvas">
        <div className="mx-auto grid max-w-6xl gap-6 px-4 py-10 text-sm text-ink-3 sm:px-6 md:grid-cols-[1.4fr_1fr]">
          <div className="space-y-2">
            <p className="font-medium text-ink-2">PhysiqueGap</p>
            <p>
              Not medical advice; see a professional for injuries or health conditions. The leanness estimate is
              experimental and has not been validated on real photos.
            </p>
          </div>
          <div className="flex gap-6 md:justify-end">
            <Link to="/guide" className="hover:text-ink">
              Guide
            </Link>
            <Link to="/privacy" className="hover:text-ink">
              Privacy
            </Link>
            <Link to="/app" className="hover:text-ink">
              Start scan
            </Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
