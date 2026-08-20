import { createHash } from "node:crypto";
import fs from "node:fs";
import { expect, test } from "@playwright/test";

test("directional NIfTI can be viewed, annotated, resumed, finalized, and exported", async ({
  page,
}) => {
  const fixture = process.env.ETHMOID_E2E_FIXTURE;
  if (!fixture || !fs.existsSync(fixture)) throw new Error("Synthetic E2E fixture is unavailable.");

  await page.goto("/");
  await page.getByRole("button", { name: "Import scan" }).last().click();
  await page.getByRole("textbox", { name: /Neutral case label/ }).fill("Case901");
  await page.getByLabel(/NIfTI volume/).setInputFiles(fixture);
  await page.getByRole("button", { name: "Import scan" }).last().click();

  await expect(page.getByRole("heading", { name: "Case901" })).toBeVisible();
  await expect(page.getByText("UNVERIFIED", { exact: true })).toBeVisible();
  await expect(page.getByText("R · A · S", { exact: true })).toBeVisible();
  const canvas = page.getByLabel("Linked axial, coronal, sagittal, and 3D CT viewer");
  await expect(canvas).toBeVisible();
  await expect(page.locator(".viewer-error")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Save", exact: true })).toBeEnabled();
  await expect(page.getByText(/^Voxel: (?!—)/)).toBeVisible();

  const multiplanarImage = await canvas.screenshot();
  expect(multiplanarImage.byteLength).toBeGreaterThan(5_000);
  const multiplanarHash = createHash("sha256").update(multiplanarImage).digest("hex");

  await page.getByRole("button", { name: "3D", exact: true }).click();
  await expect(page.getByRole("button", { name: "3D", exact: true })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await page.waitForTimeout(1_000);
  const renderImage = await canvas.screenshot();
  expect(createHash("sha256").update(renderImage).digest("hex")).not.toBe(multiplanarHash);

  await page.getByRole("button", { name: "Axial · Coronal · Sagittal" }).click();
  await page.getByRole("button", { name: "Draw", exact: true }).click();
  const box = await canvas.boundingBox();
  if (!box) throw new Error("Viewer canvas has no drawable bounds.");
  const start = { x: box.x + box.width * 0.28, y: box.y + box.height * 0.72 };
  const end = { x: start.x + 18, y: start.y + 6 };
  await page.mouse.move(start.x, start.y);
  await page.mouse.down();
  await page.mouse.move(end.x, end.y, { steps: 6 });
  await page.mouse.up();

  await expect(page.getByText("Saved revision 1", { exact: true })).toBeVisible({
    timeout: 30_000,
  });
  await expect(page.getByRole("button", { name: "Undo" })).toBeEnabled();
  await expect(page.getByText(/[1-9][0-9]* voxels/).first()).toBeVisible();

  await page.getByRole("button", { name: "3D", exact: true }).click();
  await page.waitForTimeout(1_000);
  await page.reload();
  await expect(page.getByRole("button", { name: "3D", exact: true })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(page.getByText("1", { exact: true }).last()).toBeVisible();
  await expect(page.locator(".viewer-error")).toHaveCount(0);

  // Browser Back must flush review and viewer state instead of discarding
  // local controls during workspace unmount. The unit suite also exercises
  // the same guard with an annotation change pending.
  await page.getByRole("textbox", { name: "Reviewer notes" }).fill("Browser-back persistence check");
  await page.goBack();
  await expect(page.getByRole("heading", { name: "Ethmoid annotation" })).toBeVisible();
  await page.getByRole("button", { name: "Continue annotation" }).click();
  await expect(page.locator(".metadata-grid").getByText("1", { exact: true })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Reviewer notes" })).toHaveValue(
    "Browser-back persistence check",
  );

  for (const label of [
    "Complete ethmoid coverage",
    "Orientation is correct",
    "Image quality is acceptable",
    "Slice continuity is intact",
  ]) {
    await page.getByRole("checkbox", { name: label }).check();
  }
  await page.getByRole("button", { name: "Finalize for review" }).click();
  await expect(page.getByText("Finalized for review", { exact: true })).toBeVisible();
  await expect(page.locator("header").getByText("Ready for review", { exact: true })).toBeVisible();

  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("link", { name: "Export" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(/^Case901_annotation\.zip$/);
  expect(await download.path()).toBeTruthy();
  await expect(page.getByText(/does not calculate a complexity score/i)).toBeVisible();
});
