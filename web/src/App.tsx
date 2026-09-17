import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { Dashboard } from "./pages/Dashboard";

const Workspace = lazy(() =>
  import("./pages/CountingWorkspace").then((module) => ({ default: module.CountingWorkspace })),
);

type Route = { page: "dashboard" } | { page: "workspace"; caseId: string };
type NavigationGuard = () => Promise<boolean>;

function parseRoute(): Route {
  const match = window.location.pathname.match(/^\/cases\/([^/]+)$/);
  return match
    ? { page: "workspace", caseId: decodeURIComponent(match[1]) }
    : { page: "dashboard" };
}

export function App() {
  const [route, setRoute] = useState<Route>(parseRoute);
  const routeRef = useRef(route);
  const navigationGuardRef = useRef<NavigationGuard | null>(null);
  const navigationInFlightRef = useRef(false);
  routeRef.current = route;

  const registerNavigationGuard = useCallback((guard: NavigationGuard | null) => {
    navigationGuardRef.current = guard;
  }, []);

  useEffect(() => {
    const onPopState = () => {
      const next = parseRoute();
      const guard = navigationGuardRef.current;
      if (!guard) {
        setRoute(next);
        return;
      }
      if (navigationInFlightRef.current) return;
      navigationInFlightRef.current = true;
      void guard()
        .then((allowed) => {
          if (allowed) {
            setRoute(next);
            return;
          }
          const current = routeRef.current;
          const currentPath =
            current.page === "dashboard"
              ? "/"
              : `/cases/${encodeURIComponent(current.caseId)}`;
          window.history.pushState({}, "", currentPath);
        })
        .finally(() => {
          navigationInFlightRef.current = false;
        });
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const navigate = (next: Route) => {
    const path = next.page === "dashboard" ? "/" : `/cases/${encodeURIComponent(next.caseId)}`;
    window.history.pushState({}, "", path);
    setRoute(next);
  };

  if (route.page === "workspace") {
    return (
      <Suspense fallback={<main className="center-state">Opening counting workspace…</main>}>
        <Workspace
          key={route.caseId}
          caseId={route.caseId}
          onBack={() => navigate({ page: "dashboard" })}
          registerNavigationGuard={registerNavigationGuard}
        />
      </Suspense>
    );
  }

  return <Dashboard onOpenCase={(caseId) => navigate({ page: "workspace", caseId })} />;
}
