import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export default async function globalTeardown(): Promise<void> {
  const root = process.env.ETHMOID_E2E_ROOT;
  if (!root) return;
  const resolved = path.resolve(root);
  const expectedParent = path.resolve(os.tmpdir());
  if (path.dirname(resolved) !== expectedParent || !path.basename(resolved).startsWith("ethmoid-e2e-")) {
    throw new Error("Refusing to remove an unexpected E2E directory.");
  }
  const response = await fetch(`http://127.0.0.1:8766/__e2e__/release-workers?root=${encodeURIComponent(path.basename(resolved))}`, { method: "POST", signal: AbortSignal.timeout(15_000) });
  if (!response.ok || (await response.json()).work_root !== path.basename(resolved)) {
    throw new Error("Synthetic server workers could not be released safely; test files retained.");
  }
  fs.rmSync(resolved, { recursive: true, force: true, maxRetries: 5, retryDelay: 250 });
}
