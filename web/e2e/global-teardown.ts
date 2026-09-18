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
  const released = await response.json();
  if (!response.ok || released.work_root !== path.basename(resolved) || !Number.isInteger(released.process_id) || released.process_id <= 0 || released.process_id === process.pid) {
    throw new Error("Synthetic server workers could not be released safely; test files retained.");
  }
  // A stopped worker is not enough: in-flight FileResponses and SQLite may
  // still hold Windows handles until the isolated server actually exits.
  const deadline = Date.now() + 15_000;
  while (true) {
    try { process.kill(released.process_id, 0); } // Existence probe only; sends no signal.
    catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ESRCH") break;
      throw error;
    }
    if (Date.now() >= deadline) throw new Error("Synthetic server has not exited; test files retained.");
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  await fs.promises.rm(resolved, { recursive: true, force: true, maxRetries: 5, retryDelay: 250 });
}
