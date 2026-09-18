import { describe, expect, it } from "vitest";
import { instanceRgba } from "./instanceOverlay";

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
});
