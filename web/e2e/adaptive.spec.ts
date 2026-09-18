import { expect, test } from "@playwright/test";
import fs from "node:fs";

test("synthetic upload client contract resumes checked chunks and retains every study outcome", async ({ page }) => {
  const size = 9 * 1024 * 1024;
  const payload = Buffer.alloc(size, 7);
  let received = 0; let interrupted = false; let finalized = false; let creations = 0;
  const offsets: number[] = [];
  const session = () => ({ upload_id: "synthetic-session", status: finalized ? "completed" : "uploading",
    files: [{ index: 0, name: "synthetic-study.zip", size, received_bytes: received, status: received === size ? "archive" : "uploading" }],
    studies: finalized ? [{ study_id: "upload-study-1", study_label: "Case970", series_count: 3, preparation_status: "ready", analysis_status: "queued" }, { study_id: "upload-study-2", study_label: "Case971", series_count: 2, preparation_status: "unavailable", analysis_status: "unsupported_input", reason_code: "SCOUT_ONLY" }] : [],
    issues: finalized ? [{ item_index: 4, reason_code: "ANCILLARY_FILE" }] : [], limits: { max_expanded_bytes: 20 * 1024 ** 3, max_files: 50000 } });
  await page.addInitScript(() => localStorage.setItem("ethmoid:first-run-guide", "2"));
  await page.route("**/api/uploads**", async route => {
    const request = route.request(); const url = new URL(request.url());
    if (request.method() === "POST" && url.pathname === "/api/uploads") {
      creations++; expect(request.postDataJSON()).toEqual({ files: [{ name: "synthetic-study.zip", size }] });
      await route.fulfill({ json: session() }); return;
    }
    if (request.method() === "PUT") {
      const offset = Number(url.searchParams.get("offset")); offsets.push(offset);
      const bytes = request.postDataBuffer()!;
      expect(bytes.equals(payload.subarray(offset, offset + bytes.length))).toBe(true);
      received = Math.max(received, offset + bytes.length);
      if (!interrupted) { interrupted = true; await route.fulfill({ status: 503, json: { detail: "Synthetic transfer interruption. Resume to continue." } }); return; }
      await route.fulfill({ json: session().files[0] }); return;
    }
    if (url.pathname.endsWith("/finalize")) { expect(received).toBe(size); finalized = true; }
    await route.fulfill({ json: session() });
  });
  await page.goto("/"); await page.getByRole("button", { name: "Import scan", exact: true }).first().click();
  await page.getByRole("radio", { name: "DICOM ZIP", exact: true }).click();
  await page.getByLabel("DICOM ZIP archive").setInputFiles({ name: "synthetic-study.zip", mimeType: "application/zip", buffer: payload });
  await page.getByRole("button", { name: "Import scan", exact: true }).last().click();
  await expect(page.getByRole("alert")).toHaveText("Synthetic transfer interruption. Resume to continue.");
  await page.reload(); await page.getByRole("button", { name: "Import scan", exact: true }).first().click();
  await expect(page.getByRole("button", { name: "Resume upload" })).toBeVisible();
  await page.getByRole("button", { name: "Start a separate upload" }).click();
  await expect(page.getByRole("button", { name: "Import scan", exact: true }).last()).toBeEnabled();
  await page.getByLabel("Saved uploads", { exact: true }).selectOption("synthetic-session");
  await expect(page.getByRole("button", { name: "Resume upload" })).toBeVisible();
  await page.getByRole("radio", { name: "DICOM ZIP", exact: true }).click();
  await page.getByLabel("DICOM ZIP archive").setInputFiles({ name: "synthetic-study.zip", mimeType: "application/zip", buffer: payload });
  await page.getByRole("button", { name: "Resume upload" }).click();
  await expect(page.getByText("Upload cataloged", { exact: true })).toBeVisible();
  await expect(page.getByText("Case970 · 3 series")).toBeVisible(); await expect(page.getByText("Case971 · 2 series")).toBeVisible();
  await expect(page.getByText("Analysis: queued")).toBeVisible(); await expect(page.getByText("Analysis: unsupported input")).toBeVisible();
  expect(offsets).toEqual([0, 0, 8 * 1024 * 1024]); expect(creations).toBe(1);
  await page.getByText("File outcomes (1)").click(); await expect(page.getByText("Item 5: ancillary file")).toBeVisible();
});

test("synthetic native regions save, produce an assisted child, preserve its parent and export after reload", async ({ page }) => {
  const errors: string[] = []; page.on("pageerror", error => errors.push(error.message));
  const fixture = process.env.ETHMOID_E2E_FIXTURE!;
  await page.addInitScript(() => localStorage.setItem("ethmoid:first-run-guide", "2"));
  const imported = await page.request.post("/api/imports/nifti", { multipart: { case_label: "Case950", file: { name: "synthetic.nii", mimeType: "application/octet-stream", buffer: fs.readFileSync(fixture) } } });
  expect(imported.ok(), await imported.text()).toBe(true);
  const caseId = (await imported.json()).case.case_id;
  await page.goto(`/cases/${caseId}`);
  await page.getByRole("button", { name: "Run automatic count" }).click();
  await expect(page.getByTestId("total-count")).toHaveText("2");
  await expect(page.getByRole("button", { name: "Review anatomical boundaries" })).toBeEnabled();
  const parent = (await (await page.request.get(`/api/cases/${caseId}/count-runs`)).json()).runs[0].run_id;
  await page.getByRole("button", { name: "Review anatomical boundaries" }).click();
  await expect(page.getByRole("combobox", { name: "Anatomical region" })).toBeVisible();
  await page.getByRole("combobox", { name: "Anatomical region" }).selectOption("2");
  await page.getByLabel("Boundary review note").fill("Synthetic bilateral region review");
  await page.getByRole("button", { name: "Save anatomical boundaries" }).click();
  await expect(page.getByText(/Anatomical boundary saved/)).toBeVisible();
  const saved = (await (await page.request.get(`/api/cases/${caseId}/count-runs/${parent}/boundaries`)).json()).boundaries[0];
  await page.reload();
  await page.getByRole("button", { name: "Review anatomical boundaries" }).click();
  await expect(page.getByLabel("Saved anatomical boundaries")).toHaveValue(saved.boundary_id);
  await expect(page.getByLabel("Boundary review note")).toHaveValue("Synthetic bilateral region review");
  await expect(page.getByRole("button", { name: "Save anatomical boundaries" })).toBeDisabled();
  await page.getByRole("button", { name: "Run assisted count" }).click();
  await expect(page.getByText("Assisted result", { exact: true })).toBeVisible();
  await expect(page.getByTestId("total-count")).toHaveText("2");
  const histories = (await (await page.request.get(`/api/cases/${caseId}/count-runs`)).json()).runs;
  expect(histories).toHaveLength(2); expect(histories[0].assistance.parent_run_id).toBe(parent);
  expect(histories[0].outcome).toBe("assisted_candidate_count");
  await page.reload(); await expect(page.getByText("Assisted result", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Review parent attempt" }).click();
  await expect(page.getByText("Automatic result", { exact: true })).toBeVisible();
  await expect(page.getByTestId("total-count")).toHaveText("2");
  await page.getByRole("combobox", { name: "Run history" }).selectOption(histories[0].run_id);
  const downloadPromise = page.waitForEvent("download"); await page.getByRole("link", { name: "Export revision 0" }).click();
  const download = await downloadPromise; expect(await download.failure()).toBeNull();
  expect(fs.readFileSync((await download.path())!).subarray(0, 2).toString()).toBe("PK");
  await page.getByRole("button", { name: /#300 · right/ }).click();
  await page.getByRole("button", { name: "Mark reviewed" }).click();
  await expect(page.getByRole("link", { name: "Export revision 1" })).toBeVisible();
  await page.getByRole("button", { name: /#300 · right/ }).click();
  await page.getByLabel("Overlay content").selectOption("cavities");
  const canvas = page.getByLabel("Linked axial, coronal, sagittal, and 3D CT viewer");
  const allCells = await canvas.screenshot();
  await page.getByLabel("Overlay content").selectOption("uncertain");
  await expect(page.getByText(/1 included cell needs review/)).toBeVisible();
  await expect.poll(async () => (await canvas.screenshot()).equals(allCells)).toBe(false);
  await expect(page.getByTestId("total-count")).toHaveText("2");
  await expect(page.locator(".count-coordinates")).toContainText("Cell #300");
  expect(errors).toEqual([]);
});
