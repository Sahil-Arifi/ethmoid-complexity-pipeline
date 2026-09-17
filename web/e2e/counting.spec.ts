import { expect, test } from "@playwright/test";
import fs from "node:fs";

test("automatic count, uint16 overlay, corrections, original result, reload, and export", async ({ page }, testInfo) => {
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
});
