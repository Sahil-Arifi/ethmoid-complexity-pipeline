import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

export default function globalSetup(): void {
  const root = process.env.ETHMOID_E2E_ROOT;
  const fixture = process.env.ETHMOID_E2E_FIXTURE;
  if (!root || !fixture) throw new Error("E2E temporary paths were not configured.");
  fs.mkdirSync(path.dirname(fixture), { recursive: true });
  const python = path.resolve(process.cwd(), "..", ".venv", "Scripts", "python.exe");
  const generator = path.resolve(process.cwd(), "..", "scripts", "create_e2e_fixture.py");
  // The synthetic server creates this same fixture before seeding its catalog.
  // Do not rewrite a NIfTI that the running server may currently have mapped.
  if (!fs.existsSync(fixture)) execFileSync(python, [generator, fixture], { stdio: "inherit" });
}
