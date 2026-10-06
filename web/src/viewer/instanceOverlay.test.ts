import { describe, expect, it } from "vitest";
import { instanceColor, instanceRgba } from "./instanceOverlay";

describe("cell review overlay", () => {
  it("filters review flags without modifying native IDs, including IDs above 255", () => {
    const native = new Uint16Array([0, 1, 300, 301, 65535]);
    const original = native.slice();
    const all = instanceRgba(native);
    const review = instanceRgba(native, new Set([300, 65535]));
    expect([...review.filter((_, index) => index % 4 === 3)]).toEqual([0, 0, 255, 0, 255]);
    expect(review.slice(8, 12)).toEqual(all.slice(8, 12));
    expect(review.slice(16, 20)).toEqual(all.slice(16, 20));
    expect(native).toEqual(original);
    expect(all.slice(0, 4)).toEqual(new Uint8Array(4));
  });

  it("hides every voxel when no included cells need review", () => {
    expect(instanceRgba(new Uint16Array([0, 3, 300]), new Set())).toEqual(new Uint8Array(12));
  });

  it("highlights a high native ID, dims its neighbors, and keeps background transparent", () => {
    const native = new Uint16Array([0, 300, 301, 65535]);
    const original = native.slice();
    const focused = instanceRgba(native, undefined, 300);
    expect(focused.slice(0, 4)).toEqual(new Uint8Array(4));
    expect([...focused.slice(4, 8)]).toEqual([255, 245, 92, 255]);
    expect([...focused.filter((_, index) => index % 4 === 3)]).toEqual([0, 255, 55, 55]);
    expect(instanceColor(300, true)).toBe("rgb(255, 245, 92)");
    expect(instanceColor(301)).toBe("rgb(48, 165, 219)");
    expect(instanceColor(0)).toBe("transparent");
    expect(native).toEqual(original);
  });

  it("never reveals a filtered-out candidate when focus changes", () => {
    const native = new Uint16Array([0, 300, 301, 65535]);
    const filtered = instanceRgba(native, new Set([301, 65535]), 300);
    expect([...filtered.filter((_, index) => index % 4 === 3)]).toEqual([0, 0, 55, 55]);
    const focused = instanceRgba(native, new Set([301, 65535]), 65535);
    expect([...focused.slice(12, 16)]).toEqual([255, 245, 92, 255]);
    expect(instanceRgba(native, new Set(), 300)).toEqual(new Uint8Array(16));
  });
});
