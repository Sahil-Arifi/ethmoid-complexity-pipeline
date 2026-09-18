import { defineConfig } from "@playwright/test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const runId = process.env.ETHMOID_E2E_RUN_ID ?? String(process.pid);
process.env.ETHMOID_E2E_RUN_ID = runId;
const e2eRoot = path.join(os.tmpdir(), `ethmoid-e2e-${runId}`);
const fixture = path.join(e2eRoot, "fixture", "directional-phantom.nii");
process.env.ETHMOID_E2E_ROOT = e2eRoot;
process.env.ETHMOID_E2E_FIXTURE = fixture;

// Use Playwright's clean bundled browser unless explicitly overridden. A host
// browser can inherit native integrations that intercept localhost downloads.
const browserExecutable = process.env.ETHMOID_BROWSER_EXECUTABLE;
if (browserExecutable && !fs.existsSync(browserExecutable)) {
  throw new Error("ETHMOID_BROWSER_EXECUTABLE does not identify an existing browser executable.");
}

const appExecutable = process.env.ETHMOID_E2E_PYTHON ?? path.resolve(
  process.cwd(),
  "..",
  ".venv",
  "Scripts",
  "python.exe",
);
const testServer = path.resolve(process.cwd(), "..", "scripts", "create_e2e_server.py");

export default defineConfig({
  testDir: "./e2e",
  workers: 1,
  fullyParallel: false,
  timeout: 120_000,
  expect: { timeout: 20_000 },
  outputDir: path.join(e2eRoot, "test-results"),
  globalSetup: "./e2e/global-setup.ts",
  globalTeardown: "./e2e/global-teardown.ts",
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:8766",
    headless: true,
    viewport: { width: 1280, height: 900 },
    launchOptions: {
      ...(browserExecutable ? { executablePath: browserExecutable } : {}),
      args: ["--enable-webgl", "--ignore-gpu-blocklist", "--use-angle=swiftshader"],
    },
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  webServer: {
    command: `"${appExecutable}" "${testServer}" --work-root "${e2eRoot}"`,
    url: "http://127.0.0.1:8766/api/health",
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
