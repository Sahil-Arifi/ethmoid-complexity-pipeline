import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import type { Niivue } from "@niivue/niivue";
import { api } from "../api/client";
import type { ViewerState } from "../api/types";
import { annotationLabels, type ProtocolDefinition } from "../config/protocol";
import { instanceRgba } from "./instanceOverlay";

export type ViewerTool = "navigate" | "window" | "draw" | "erase" | "fill";
export type ViewMode = "multiplanar" | "render";

export type ViewerLocation = {
  voxel: string;
  millimeters: string;
  cellId?: number;
};

export type NiiVueViewerHandle = {
  exportMask: () => Promise<Blob>;
  getViewerState: () => ViewerState;
  undo: () => void;
  redo: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;
  resetView: () => void;
  focusWorld: (point: [number, number, number]) => void;
  getWorldPosition: () => [number, number, number];
  clearSelectionMask: () => void;
  prepareCellMask: (url: string, id: number) => Promise<void>;
  prepareRegionMask: (url: string) => Promise<void>;
};

type Props = {
  caseId: string;
  volumeFileType: "nii" | "nii.gz";
  protocol: ProtocolDefinition;
  hasAnnotation: boolean;
  initialState?: ViewerState | null;
  tool: ViewerTool;
  selectedLabel: number;
  brushSize: number;
  overlayOpacity: number;
  overlayVisible: boolean;
  viewMode: ViewMode;
  windowCenter: number;
  windowWidth: number;
  onReady: () => void;
  onError: (message: string) => void;
  onDirty: () => void;
  onViewStateChange: () => void;
  onWindowChange: (center: number, width: number) => void;
  onLocationChange: (location: ViewerLocation) => void;
  onHistoryChange: (canUndo: boolean, canRedo: boolean) => void;
  instanceOverlayUrl?: string;
  visibleCellIds?: number[];
  focusedCellId?: number;
};

type LocationPayload = {
  mm?: ArrayLike<number>;
  vox?: ArrayLike<number>;
};

type DrawingHistory = {
  snapshots: Uint8Array[];
  index: number;
};

const emptyHistory = (): DrawingHistory => ({ snapshots: [], index: -1 });

function hexToRgb(hex: string): [number, number, number] {
  return [
    Number.parseInt(hex.slice(1, 3), 16),
    Number.parseInt(hex.slice(3, 5), 16),
    Number.parseInt(hex.slice(5, 7), 16),
  ];
}

function formatTriplet(values: ArrayLike<number> | undefined, digits: number): string {
  if (!values || values.length < 3) return "—";
  return Array.from(values)
    .slice(0, 3)
    .map((value) => Number(value).toFixed(digits))
    .join(", ");
}

function readWindow(nv: Niivue): { center: number; width: number } | null {
  const volume = nv.volumes[0];
  const minimum = Number(volume?.cal_min);
  const maximum = Number(volume?.cal_max);
  const width = maximum - minimum;
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || width <= 0) return null;
  return { center: (minimum + maximum) / 2, width };
}

export const NiiVueViewer = forwardRef<NiiVueViewerHandle, Props>(
  function NiiVueViewer(
    {
      caseId,
      volumeFileType,
      protocol,
      hasAnnotation,
      initialState,
      tool,
      selectedLabel,
      brushSize,
      overlayOpacity,
      overlayVisible,
      viewMode,
      windowCenter,
      windowWidth,
      onReady,
      onError,
      onDirty,
      onViewStateChange,
      onWindowChange,
      onLocationChange,
      onHistoryChange,
      instanceOverlayUrl,
      visibleCellIds,
      focusedCellId,
    },
    ref,
  ) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const niivueRef = useRef<Niivue | null>(null);
    const instanceLabelsRef = useRef<import("@niivue/niivue").NVImage | null>(null);
    const instanceDisplayRef = useRef<import("@niivue/niivue").NVImage | null>(null);
    const historyRef = useRef<DrawingHistory>(emptyHistory());
    const initialStateRef = useRef(initialState);
    const initialHasAnnotationRef = useRef(hasAnnotation);
    const callbacksRef = useRef({
      onReady,
      onError,
      onDirty,
      onViewStateChange,
      onWindowChange,
      onLocationChange,
      onHistoryChange,
    });
    const controlStateRef = useRef({
      tool,
      selectedLabel,
      brushSize,
      overlayOpacity,
      overlayVisible,
      viewMode,
      windowCenter,
      windowWidth,
    });
    const [loading, setLoading] = useState(true);
    const cellFilterKey = visibleCellIds?.join(",");
    const overlayControlsRef = useRef({ cellFilterKey, focusedCellId });
    overlayControlsRef.current = { cellFilterKey, focusedCellId };

    callbacksRef.current = {
      onReady,
      onError,
      onDirty,
      onViewStateChange,
      onWindowChange,
      onLocationChange,
      onHistoryChange,
    };
    controlStateRef.current = {
      tool,
      selectedLabel,
      brushSize,
      overlayOpacity,
      overlayVisible,
      viewMode,
      windowCenter,
      windowWidth,
    };

    const reportHistory = () => {
      const history = historyRef.current;
      callbacksRef.current.onHistoryChange(
        history.index > 0,
        history.index >= 0 && history.index < history.snapshots.length - 1,
      );
    };

    const addHistorySnapshot = (bitmap: Uint8Array | null) => {
      if (!bitmap) return;
      const history = historyRef.current;
      history.snapshots = history.snapshots.slice(0, history.index + 1);
      history.snapshots.push(bitmap.slice());
      if (history.snapshots.length > 30) history.snapshots.shift();
      history.index = history.snapshots.length - 1;
      reportHistory();
    };

    const restoreHistorySnapshot = (nextIndex: number) => {
      const nv = niivueRef.current;
      const history = historyRef.current;
      const snapshot = history.snapshots[nextIndex];
      if (!nv || !snapshot) return;
      history.index = nextIndex;
      nv.drawBitmap = snapshot.slice();
      nv.refreshDrawing(true);
      callbacksRef.current.onDirty();
      reportHistory();
    };

    useImperativeHandle(
      ref,
      () => ({
        async exportMask() {
          const nv = niivueRef.current;
          if (!nv) throw new Error("The image viewer is not ready.");
          const result = await nv.saveImage({
            filename: "",
            isSaveDrawing: true,
            volumeByIndex: 0,
          });
          if (!(result instanceof Uint8Array)) {
            throw new Error("The annotation could not be serialized as NIfTI.");
          }
          const bytes = result.slice();
          return new Blob([bytes.buffer as ArrayBuffer], { type: "application/octet-stream" });
        },
        getViewerState() {
          const nv = niivueRef.current;
          const controls = controlStateRef.current;
          if (!nv) {
            return {
              view_mode: controls.viewMode,
              selected_label: controls.selectedLabel,
              brush_size: controls.brushSize,
              overlay_opacity: controls.overlayOpacity,
              active_tool: controls.tool,
              window_center: controls.windowCenter,
              window_width: controls.windowWidth,
            };
          }
          const crosshair = Array.from(nv.scene.crosshairPos).slice(0, 3) as [
            number,
            number,
            number,
          ];
          const panZoom = Array.from(nv.scene.pan2Dxyzmm).slice(0, 4) as [
            number,
            number,
            number,
            number,
          ];
          const actualWindow = readWindow(nv);
          return {
            crosshair,
            pan_zoom: panZoom,
            view_mode: controls.viewMode,
            selected_label: controls.selectedLabel,
            brush_size: controls.brushSize,
            overlay_opacity: controls.overlayOpacity,
            active_tool: controls.tool,
            window_center: actualWindow?.center ?? controls.windowCenter,
            window_width: actualWindow?.width ?? controls.windowWidth,
            render_azimuth: nv.scene.renderAzimuth,
            render_elevation: nv.scene.renderElevation,
            volume_scale: nv.volScaleMultiplier,
          };
        },
        undo() {
          const history = historyRef.current;
          if (history.index > 0) restoreHistorySnapshot(history.index - 1);
        },
        redo() {
          const history = historyRef.current;
          if (history.index < history.snapshots.length - 1) {
            restoreHistorySnapshot(history.index + 1);
          }
        },
        canUndo() {
          return historyRef.current.index > 0;
        },
        canRedo() {
          const history = historyRef.current;
          return history.index >= 0 && history.index < history.snapshots.length - 1;
        },
        resetView() {
          const nv = niivueRef.current;
          if (!nv) return;
          nv.scene.crosshairPos = new Float32Array([0.5, 0.5, 0.5]);
          nv.scene.pan2Dxyzmm = new Float32Array([0, 0, 0, 1]);
          nv.volScaleMultiplier = 1;
          nv.setRenderAzimuthElevation(110, 10);
          nv.drawScene();
          callbacksRef.current.onViewStateChange();
        },
        focusWorld(point) {
          const nv = niivueRef.current;
          if (!nv) return;
          nv.scene.crosshairPos = new Float32Array(nv.mm2frac(point));
          nv.drawScene();
          nv.createOnLocationChange();
        },
        getWorldPosition() {
          const nv = niivueRef.current;
          if (!nv) throw new Error("The image viewer is not ready.");
          return Array.from(nv.frac2mm(nv.scene.crosshairPos)).slice(0, 3) as [number, number, number];
        },
        clearSelectionMask() {
          const nv = niivueRef.current;
          if (!nv) return;
          nv.createEmptyDrawing();
          historyRef.current = emptyHistory();
          addHistorySnapshot(nv.drawBitmap);
          nv.drawScene();
        },
        async prepareCellMask(url, id) {
          const nv = niivueRef.current;
          if (!nv) throw new Error("The image viewer is not ready.");
          const { NVImage } = await import("@niivue/niivue");
          const mask = await NVImage.loadFromUrl({ url, name: "cavities.nii.gz" });
          if (niivueRef.current !== nv) return;
          if (!mask.img || !mask.hdr) throw new Error("The cell mask could not be opened.");
          // Canonical instance IDs remain uint16 in the overlay. Only the
          // selected cell becomes a binary editable drawing; loadDrawing
          // handles the source-to-display orientation transformation.
          mask.img = Uint8Array.from(mask.img, value => value === id ? 1 : 0);
          mask.hdr.datatypeCode = 2;
          mask.hdr.numBitsPerVoxel = 8;
          mask.hdr.scl_slope = 1;
          mask.hdr.scl_inter = 0;
          if (!nv.loadDrawing(mask)) throw new Error("The cell mask geometry does not match this CT.");
          historyRef.current = emptyHistory();
          addHistorySnapshot(nv.drawBitmap);
        },
        async prepareRegionMask(url) {
          const nv = niivueRef.current;
          if (!nv) throw new Error("The image viewer is not ready.");
          const { NVImage } = await import("@niivue/niivue");
          const mask = await NVImage.loadFromUrl({ url, name: "boundary.nii.gz" });
          if (niivueRef.current !== nv) return;
          if (!mask.img || !mask.hdr) throw new Error("The anatomical boundary mask could not be opened.");
          for (const value of mask.img) {
            if (value !== 0 && value !== 1 && value !== 2) throw new Error("Anatomical regions must use labels 1 for left and 2 for right.");
          }
          mask.img = Uint8Array.from(mask.img);
          mask.hdr.datatypeCode = 2; mask.hdr.numBitsPerVoxel = 8;
          mask.hdr.scl_slope = 1; mask.hdr.scl_inter = 0;
          if (!nv.loadDrawing(mask)) throw new Error("The anatomical boundary geometry does not match this CT.");
          historyRef.current = emptyHistory(); addHistorySnapshot(nv.drawBitmap);
        },
      }),
      [],
    );

    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      let cancelled = false;
      let instance: Niivue | null = null;
      const notifyViewStateAfterInteraction = () => {
        window.setTimeout(() => {
          const nv = niivueRef.current;
          const actualWindow = nv ? readWindow(nv) : null;
          if (actualWindow) {
            callbacksRef.current.onWindowChange(actualWindow.center, actualWindow.width);
          }
          callbacksRef.current.onViewStateChange();
        }, 0);
      };
      canvas.addEventListener("pointerup", notifyViewStateAfterInteraction);
      canvas.addEventListener("wheel", notifyViewStateAfterInteraction, { passive: true });

      const initialize = async () => {
        try {
          setLoading(true);
          const { Niivue: NiivueClass, cmapper } = await import("@niivue/niivue");
          if (cancelled) return;
          const nv = new NiivueClass({
            backColor: [0.02, 0.025, 0.035, 1],
            crosshairColor: [0.2, 0.9, 0.95, 1],
            crosshairWidth: 1,
            dragAndDropEnabled: false,
            isRadiologicalConvention: true,
            isOrientationTextVisible: true,
            showAllOrientationMarkers: true,
            show3Dcrosshair: true,
            drawingEnabled: true,
            maxDrawUndoBitmaps: 30,
            loadingText: "Loading CT…",
            logLevel: "error",
          });
          instance = nv;
          niivueRef.current = nv;
          await nv.attachToCanvas(canvas);
          if (cancelled) return;
          await nv.loadVolumes([
            {
              url: api.volumeUrl(caseId),
              // NiiVue infers the decoder from the supplied name. The API URL
              // is intentionally extensionless, so preserve the stored format
              // explicitly for both uncompressed and gzipped NIfTI inputs.
              name: `CT.${volumeFileType}`,
              colormap: "gray",
              cal_min: -1200,
              cal_max: 1800,
              trustCalMinMax: true,
            },
          ]);
          if (cancelled) return;

          if (initialHasAnnotationRef.current) {
            const loaded = await nv.loadDrawingFromUrl(api.latestAnnotationUrl(caseId));
            if (!loaded) throw new Error("The saved annotation does not match this CT volume.");
          } else {
            nv.createEmptyDrawing();
          }

          const labels = annotationLabels(protocol);
          const labelColors = labels.map((label) => hexToRgb(label.color));
          const protocolColorMap = {
            R: [0, ...labelColors.map((color) => color[0])],
            G: [0, ...labelColors.map((color) => color[1])],
            B: [0, ...labelColors.map((color) => color[2])],
            A: [0, 255, 255, 255, 255],
            I: [0, 1, 2, 3, 4],
            labels: ["Background", ...labels.map((label) => label.name)],
          };
          nv.addColormap(protocol.protocol_version, protocolColorMap);
          // NiiVue's public setter initializes the drawing texture; its 0.69.0
          // custom-name lookup does not include runtime-added drawing maps, so
          // apply the exact protocol LUT immediately afterward.
          nv.setDrawColormap("$itksnap");
          nv.drawLut = cmapper.makeDrawLut(protocolColorMap);
          const controls = controlStateRef.current;
          const volume = nv.volumes[0];
          if (volume && controls.windowWidth > 0) {
            volume.cal_min = controls.windowCenter - controls.windowWidth / 2;
            volume.cal_max = controls.windowCenter + controls.windowWidth / 2;
          }
          nv.updateGLVolume();

          const saved = initialStateRef.current;
          if (saved?.crosshair?.length === 3) {
            nv.scene.crosshairPos = new Float32Array(saved.crosshair);
          }
          if (saved?.pan_zoom?.length === 4) {
            nv.scene.pan2Dxyzmm = new Float32Array(saved.pan_zoom);
          }
          if (typeof saved?.render_azimuth === "number") {
            nv.scene.renderAzimuth = saved.render_azimuth;
          }
          if (typeof saved?.render_elevation === "number") {
            nv.scene.renderElevation = saved.render_elevation;
          }
          if (typeof saved?.volume_scale === "number" && saved.volume_scale > 0) {
            nv.volScaleMultiplier = saved.volume_scale;
          }

          nv.opts.penSize = controls.brushSize;
          nv.setDrawOpacity(controls.overlayOpacity);
          nv.setSliceType(
            controls.viewMode === "render" ? nv.sliceTypeRender : nv.sliceTypeMultiplanar,
          );
          nv.setDrawingEnabled(controls.tool !== "navigate" && controls.tool !== "window");
          nv.setPenValue(
            controls.tool === "erase" ? 0 : controls.selectedLabel,
            controls.tool === "fill",
          );
          nv.onLocationChange = (payload) => {
            const location = payload as LocationPayload;
            const labels = instanceLabelsRef.current;
            const overlayVoxel = location.mm && labels ? labels.mm2vox(Array.from(location.mm)) : null;
            const cellId = labels && overlayVoxel ? Math.round(labels.getValue(overlayVoxel[0], overlayVoxel[1], overlayVoxel[2])) : 0;
            callbacksRef.current.onLocationChange({
              voxel: formatTriplet(location.vox, 0),
              millimeters: formatTriplet(location.mm, 1),
              cellId: Number.isFinite(cellId) && cellId > 0 ? cellId : undefined,
            });
          };
          nv.onAzimuthElevationChange = () => {
            callbacksRef.current.onViewStateChange();
          };
          nv.onDrawingChanged = (action) => {
            if (action === "draw") {
              addHistorySnapshot(nv.drawBitmap);
              callbacksRef.current.onDirty();
            }
          };
          historyRef.current = emptyHistory();
          addHistorySnapshot(nv.drawBitmap);
          nv.drawScene();
          const initialFraction = Array.from(nv.scene.crosshairPos).slice(0, 3) as [
            number,
            number,
            number,
          ];
          callbacksRef.current.onLocationChange({
            voxel: formatTriplet(nv.frac2vox(initialFraction), 0),
            millimeters: formatTriplet(nv.frac2mm(initialFraction), 1),
          });
          setLoading(false);
          callbacksRef.current.onReady();
        } catch (error) {
          if (cancelled) return;
          setLoading(false);
          const message = error instanceof Error ? error.message : "The CT viewer failed to load.";
          callbacksRef.current.onError(message);
        }
      };

      void initialize();
      return () => {
        cancelled = true;
        canvas.removeEventListener("pointerup", notifyViewStateAfterInteraction);
        canvas.removeEventListener("wheel", notifyViewStateAfterInteraction);
        instance?.cleanup();
        if (niivueRef.current === instance) {
          niivueRef.current = null;
          instanceLabelsRef.current = null;
          instanceDisplayRef.current = null;
        }
      };
    }, [caseId, protocol, volumeFileType]);

    useEffect(() => {
      const nv = niivueRef.current;
      if (!nv || loading) return;
      let cancelled = false;
      let overlay: import("@niivue/niivue").NVImage | undefined;
      const load = async () => {
        try {
          if (!instanceOverlayUrl) return;
          const { NVImage } = await import("@niivue/niivue");
          const volume = await NVImage.loadFromUrl({ url: instanceOverlayUrl, name: "cells.nii.gz" });
          if (cancelled || niivueRef.current !== nv) return;
          if (!volume.img || !volume.hdr) throw new Error("The cell overlay could not be opened.");
          // NiiVue 0.69's indexed shader clamps low values to 2/256, which
          // paints background label 0 when a LUT contains IDs above 255.
          // Render an RGBA copy; retain the unchanged native uint16 image for
          // cell picking and fetch that canonical image for binary corrections.
          const { cellFilterKey: filter, focusedCellId: focused } = overlayControlsRef.current;
          const visibleIds = filter === undefined ? undefined : new Set(filter ? filter.split(",").map(Number) : []);
          const rgba = instanceRgba(volume.img, visibleIds, focused);
          const display = volume.clone();
          display.img = rgba;
          display.hdr!.datatypeCode = 2304; // NIfTI RGBA32, display only.
          display.hdr!.numBitsPerVoxel = 32;
          display.hdr!.scl_slope = 1;
          display.hdr!.scl_inter = 0;
          const controls = controlStateRef.current;
          display.opacity = controls.overlayVisible ? controls.overlayOpacity : 0;
          instanceLabelsRef.current = volume;
          instanceDisplayRef.current = display;
          overlay = display;
          nv.addVolume(display);
          nv.createOnLocationChange();
        } catch (error) {
          if (!cancelled) callbacksRef.current.onError(error instanceof Error ? error.message : "Cell overlay failed to load.");
        }
      };
      void load();
      return () => {
        cancelled = true;
        if (overlay && niivueRef.current === nv) {
          instanceLabelsRef.current = null;
          instanceDisplayRef.current = null;
          nv.removeVolume(overlay);
          nv.createOnLocationChange();
        }
      };
    }, [instanceOverlayUrl, loading]);

    useEffect(() => {
      const nv = niivueRef.current;
      const labels = instanceLabelsRef.current;
      const display = instanceDisplayRef.current;
      if (!nv || !labels?.img || !display || loading) return;
      const visibleIds = cellFilterKey === undefined ? undefined : new Set(cellFilterKey ? cellFilterKey.split(",").map(Number) : []);
      display.img = instanceRgba(labels.img, visibleIds, focusedCellId);
      nv.updateGLVolume();
    }, [cellFilterKey, focusedCellId, loading]);

    useEffect(() => {
      const nv = niivueRef.current;
      if (!nv) return;
      nv.opts.penSize = brushSize;
      if (tool === "navigate") {
        nv.setDrawingEnabled(false);
        nv.setDragMode("pan");
      } else if (tool === "window") {
        nv.setDrawingEnabled(false);
        nv.setDragMode("windowing");
      } else {
        nv.setDrawingEnabled(true);
        nv.setPenValue(tool === "erase" ? 0 : selectedLabel, tool === "fill");
      }
      nv.drawScene();
    }, [brushSize, selectedLabel, tool]);

    useEffect(() => {
      const nv = niivueRef.current;
      if (!nv) return;
      nv.setDrawOpacity(overlayVisible ? overlayOpacity : 0);
      if (nv.volumes.length > 1) nv.setOpacity(1, overlayVisible ? overlayOpacity : 0);
    }, [overlayOpacity, overlayVisible]);

    useEffect(() => {
      const nv = niivueRef.current;
      if (!nv) return;
      nv.setSliceType(viewMode === "render" ? nv.sliceTypeRender : nv.sliceTypeMultiplanar);
    }, [viewMode]);

    useEffect(() => {
      const nv = niivueRef.current;
      const volume = nv?.volumes[0];
      if (!nv || !volume || windowWidth <= 0) return;
      const frame = window.requestAnimationFrame(() => {
        volume.cal_min = windowCenter - windowWidth / 2;
        volume.cal_max = windowCenter + windowWidth / 2;
        nv.updateGLVolume();
      });
      return () => window.cancelAnimationFrame(frame);
    }, [windowCenter, windowWidth]);

    return (
      <div className="viewer-frame" aria-busy={loading}>
        {loading ? <div className="viewer-loading">Loading CT volume…</div> : null}
        <canvas
          ref={canvasRef}
          className="viewer-canvas"
          aria-label="Linked axial, coronal, sagittal, and 3D CT viewer"
          tabIndex={0}
        />
      </div>
    );
  },
);
