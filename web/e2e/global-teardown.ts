import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export default function globalTeardown(): void {
  const root = process.env.ETHMOID_E2E_ROOT;
  if (!root) return;
  const resolved = path.resolve(root);
  const expectedParent = path.resolve(os.tmpdir());
  if (path.dirname(resolved) !== expectedParent || !path.basename(resolved).startsWith("ethmoid-e2e-")) {
    throw new Error("Refusing to remove an unexpected E2E directory.");
  }
  fs.rmSync(resolved, { recursive: true, force: true, maxRetries: 5, retryDelay: 250 });
}
