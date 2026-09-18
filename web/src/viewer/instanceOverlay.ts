const PALETTE = [[48, 165, 219], [239, 177, 64], [103, 190, 130], [213, 125, 185], [161, 148, 220], [229, 119, 96]];

/** Display-only filtering: canonical native IDs and their measurements are never changed. */
export function instanceRgba(labels: ArrayLike<number>, visibleIds?: ReadonlySet<number>): Uint8Array {
  const rgba = new Uint8Array(labels.length * 4);
  for (let index = 0; index < labels.length; index++) {
    const id = labels[index];
    if (id === 0 || (visibleIds && !visibleIds.has(id))) continue;
    const color = PALETTE[(id - 1) % PALETTE.length];
    const offset = index * 4;
    rgba[offset] = color[0]; rgba[offset + 1] = color[1];
    rgba[offset + 2] = color[2]; rgba[offset + 3] = 255;
  }
  return rgba;
}
