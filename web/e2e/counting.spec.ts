import { expect, test } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";

test("automatic count, uint16 overlay, corrections, original result, reload, and export", async ({ page }, testInfo) => {
  const browserErrors: string[] = [];
  page.on("pageerror", error => browserErrors.push(error.message));
  const caseLabel = `Case${902 + testInfo.repeatEachIndex}`;
  const fixture = process.env.ETHMOID_E2E_FIXTURE!;
  expect(fs.existsSync(fixture)).toBeTruthy();
  await page.addInitScript(() => localStorage.setItem("ethmoid:first-run-guide", "2"));
  await page.goto("/");
  await page.getByRole("button", { name: "Import scan", exact: true }).first().click();
  await page.getByRole("textbox", { name: /Neutral case label/ }).fill(caseLabel);
  await page.getByLabel(/NIfTI volume/).setInputFiles(fixture);
  await page.getByRole("button", { name: "Import scan", exact: true }).last().click();
  await expect(page.getByRole("heading", { name: caseLabel })).toBeVisible();
  await page.getByRole("button", { name: "Run automatic count" }).click();
  await expect(page.getByTestId("total-count")).toHaveText("2");
  const canvas = page.getByLabel("Linked axial, coronal, sagittal, and 3D CT viewer");
  await expect(canvas).toBeVisible();
  await page.getByRole("button", { name: /#300 · right/ }).click();
  await expect(page.getByText("Cell #300", { exact: true }).first()).toBeVisible();
  await expect(page.locator(".count-coordinates")).toContainText("Cell #300");

  // Canonical ID 300 must render without tinting label-0 background. Compare
  // actual browser pixels, not the JavaScript LUT that the GPU may interpret.
  const shown = await canvas.screenshot();
  await page.getByRole("checkbox", { name: "Overlay", exact: true }).uncheck();
  const hidden = await canvas.screenshot();
  await page.getByRole("checkbox", { name: "Overlay", exact: true }).check();
  const pixels = JSON.parse(execFileSync(path.resolve("..", ".venv", "Scripts", "python.exe"), ["-c", `
import base64,io,json,sys
import numpy as np
from PIL import Image
shown,hidden=[np.asarray(Image.open(io.BytesIO(base64.b64decode(value))).convert('RGB')).astype(int) for value in json.load(sys.stdin)]
changed=np.max(np.abs(shown-hidden),axis=2)>2
# This directional phantom has no labelled cells in its leftmost image band.
background=(hidden.max(axis=2)<30)
background[:,max(1,hidden.shape[1]//40):]=False
print(json.dumps({'background_pixels':int(background.sum()),'changed_background':int((changed&background).sum()),'changed_pixels':int(changed.sum())}))
`], { input: JSON.stringify([shown.toString("base64"), hidden.toString("base64")]), encoding: "utf8" }));
  expect(pixels.background_pixels).toBeGreaterThan(100);
  expect(pixels.changed_background).toBe(0);
  expect(pixels.changed_pixels).toBeGreaterThan(100);

  // Exercise the RGBA display volume in the separate 3D renderer, then verify
  // the native uint16 image still supplies the crosshair's stable cell IDs.
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await expect(page.getByRole("button", { name: "3D", exact: true })).toHaveAttribute("aria-pressed", "true");
  const rendered = await canvas.screenshot();
  const renderPixels = JSON.parse(execFileSync(path.resolve("..", ".venv", "Scripts", "python.exe"), ["-c", `
import base64,io,json,sys
import numpy as np
from PIL import Image
image=np.asarray(Image.open(io.BytesIO(base64.b64decode(sys.stdin.read()))).convert('RGB'))
print(json.dumps({'visible_pixels':int((image.max(axis=2)>40).sum()),'std':float(image.std())}))
`], { input: rendered.toString("base64"), encoding: "utf8" }));
  expect(renderPixels.visible_pixels).toBeGreaterThan(1_000);
  expect(renderPixels.std).toBeGreaterThan(10);
  expect(rendered.equals(shown)).toBe(false);
  await expect(page.getByTestId("total-count")).toHaveText("2");
  await page.getByRole("button", { name: "Axial · Coronal · Sagittal" }).click();
  await expect(page.getByRole("button", { name: "Axial · Coronal · Sagittal" })).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("button", { name: /#1 · left/ }).click();
  await expect(page.locator(".count-coordinates")).toContainText("Cell #1");
  await page.getByRole("button", { name: /#300 · right/ }).click();
  await expect(page.locator(".count-coordinates")).toContainText("Cell #300");
  await expect(page.getByTestId("total-count")).toHaveText("2");
  expect(browserErrors).toEqual([]);

  // Load and serialize the canonical uint16 cell as an oriented binary drawing.
  await page.getByRole("button", { name: "Edit boundary" }).click();
  await expect(page.getByRole("heading", { name: "Edit cell #300 boundary" })).toBeVisible();
  page.once("dialog", dialog => dialog.accept());
  await page.getByRole("button", { name: "Discard correction" }).click();
  await page.getByRole("button", { name: "Remove cell" }).click();
  await expect(page.getByTestId("total-count")).toHaveText("1");
  await expect(page.getByRole("button", { name: /#300 · right/ })).toHaveCount(0);
  await page.getByRole("checkbox", { name: "Show original prediction" }).check();
  await expect(page.getByTestId("total-count")).toHaveText("2");
  await page.getByRole("checkbox", { name: "Show original prediction" }).uncheck();
  await page.getByRole("button", { name: "Add cell" }).click();
  await page.getByRole("combobox", { name: "New cell side" }).selectOption("right");
  await page.getByRole("button", { name: "Add at crosshair" }).click();
  await expect(page.getByTestId("total-count")).toHaveText("2");
  await expect(page.getByRole("button", { name: /#301 · right/ })).toBeVisible();
  await page.reload();
  await expect(page.getByTestId("total-count")).toHaveText("2");
  await expect(page.getByRole("button", { name: /#301 · right/ })).toBeVisible();
  const exportLink = page.getByRole("link", { name: "Export revision 2" });
  const exportHref = (await exportLink.getAttribute("href"))!;
  const responsePromise = page.waitForResponse(response => response.url().endsWith(exportHref)
    && response.request().resourceType() === "fetch");
  const downloadPromise = page.waitForEvent("download", { timeout: 20_000 });
  // Keep a failed response assertion as the primary error if no download starts.
  void downloadPromise.catch(() => {});
  await exportLink.click();
  const response = await responsePromise;
  expect(response.status(), response.status() === 200 ? "ZIP export" : await response.text()).toBe(200);
  expect(response.headers()["content-type"]).toBe("application/zip");
  const download = await downloadPromise;
  expect(await download.failure()).toBeNull();
  const downloaded = (await download.path())!;
  expect(downloaded).toBeTruthy();
  const bytes = fs.readFileSync(downloaded);
  expect(bytes.subarray(0, 2).toString(), bytes.subarray(0, 300).toString()).toBe("PK");
  expect(download.suggestedFilename()).toMatch(/count_revision_000002\.zip$/);
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(browserErrors).toEqual([]);
});
