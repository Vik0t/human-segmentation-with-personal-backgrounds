"use client";
import * as tf from "@tensorflow/tfjs";
import {
  GraphModel,
  loadGraphModel,
  nextFrame,
  Tensor,
  browser,
  Rank,
} from "@tensorflow/tfjs";
import {
  createContext,
  ReactNode,
  RefObject,
  useContext,
  useRef,
  useState,
} from "react";

interface TFSegmentationContextType {
  srcVideoref: RefObject<HTMLVideoElement | null>;
  canvasref: RefObject<HTMLCanvasElement | null>;
  isLoaded: boolean;
  start: () => Promise<void>;
  stop: () => void;
  setBG: (bg: string) => void;
  setJSONConfig: (jsonConfig: string) => void;
  registerErrorHandler: (handler: (error: Error) => void) => void;
  unregisterErrorHandler: (handler: (error: Error) => void) => void;
}

// TODO: add error handling

const TFSegmentationProviderContext =
  createContext<TFSegmentationContextType | null>(null);

export function useTFSegmentation() {
  const context = useContext(TFSegmentationProviderContext);
  if (!context) {
    throw new Error(
      "useTFSegmentation must be used within a TFSegmentationProvider"
    );
  }
  return context;
}

export default function TFSegmentationProvider({
  children,
}: {
  children: ReactNode;
}) {
  const srcVideoref = useRef<HTMLVideoElement>(null);
  const canvasref = useRef<HTMLCanvasElement>(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [isStarted, setIsStarted] = useState(false);
  const [jsonConfig, setJSONConfig] = useState<string | null>(null);
  const errorHandlersRef = useRef<((error: Error) => void)[]>([]);
  const stoppedRef = useRef(true);
  const modelref = useRef<GraphModel | null>(null);
  const webcamref = useRef<any>(null);
  const nextFrameref = useRef<number>(0);
  const tensorsRef = useRef<{
    r11: Tensor;
    r21: Tensor;
    r31: Tensor;
    r41: Tensor;
  } | null>(null);
  const backgroundImgref = useRef<HTMLImageElement | null>(null);

  const setBG = async (bgSrc: string) => {
    return await new Promise((resolve, reject) => {
      backgroundImgref.current = new Image();
      backgroundImgref.current.crossOrigin = "anonymous"; // For CORS if needed
      backgroundImgref.current.onload = () => {
        console.log("background image loaded");
        resolve(backgroundImgref.current);
      };
      backgroundImgref.current.onerror = () => {
        console.error("Failed to load background image");
        reject(new Error("Image load failed"));
      };
      backgroundImgref.current.src = bgSrc;
    });
  };

  const draw = async () => {
    if (
      !modelref.current ||
      !webcamref.current ||
      !canvasref.current ||
      !tensorsRef.current
    )
      return;
    await nextFrame();
    const img = await webcamref.current.capture();
    await segmentFrameWithBackground(
      img,
      backgroundImgref.current,
      modelref.current,
      canvasref.current
    );
    if (stoppedRef.current) {
      canvasref.current
        ?.getContext("2d")
        ?.clearRect(0, 0, canvasref.current.width, canvasref.current.height);
      return;
    }
    if (!isLoaded) setIsLoaded(true);
    nextFrameref.current = requestAnimationFrame(draw);
  };

  const start = async () => {
    if (isStarted || !srcVideoref.current || !canvasref.current) return;
    stoppedRef.current = false;
    setIsStarted(true);
    console.log("starting tf segmentation");
    modelref.current = await loadGraphModel("model.json");
    console.log("model loaded");
    srcVideoref.current.width = 640;
    srcVideoref.current.height = 480;
    webcamref.current = await tf.data.webcam(srcVideoref.current);
    console.log("webcam started");
    tensorsRef.current = {
      r11: tf.tensor(0),
      r21: tf.tensor(0),
      r31: tf.tensor(0),
      r41: tf.tensor(0),
    };
    if (!backgroundImgref.current) {
      await setBG("white.png");
    }
    await draw();
  };
  const stop = () => {
    if (!isStarted) return;
    stoppedRef.current = true;
    console.log("stopping tf segmentation");
    setIsStarted(false);
    cancelAnimationFrame(nextFrameref.current);
    setIsLoaded(false);
    webcamref.current.stop();
    webcamref.current = null;
    modelref.current = null;
    tensorsRef.current = null;
    nextFrameref.current = 0;
    canvasref.current
      ?.getContext("2d")
      ?.clearRect(0, 0, canvasref.current.width, canvasref.current.height);
    console.log("tf segmentation stopped");
  };
  const registerErrorHandler = (handler: (error: Error) => void) => {
    errorHandlersRef.current.push(handler);
  };
  const unregisterErrorHandler = (handler: (error: Error) => void) => {
    errorHandlersRef.current = errorHandlersRef.current.filter(
      (h: (error: Error) => void) => h !== handler
    );
  };
  return (
    <TFSegmentationProviderContext
      value={{
        srcVideoref,
        canvasref,
        isLoaded,
        start,
        stop,
        setBG,
        setJSONConfig,
        registerErrorHandler,
        unregisterErrorHandler,
      }}
    >
      {children}
    </TFSegmentationProviderContext>
  );
}

async function segmentFrameWithBackground(
  img: any,
  backgroundImg: any,
  model: GraphModel,
  canvas: HTMLCanvasElement
) {
  const src = tf.tidy(() => img.expandDims(0).div(255)); // normalize input

  // Initialize dummy recurrent states
  const [r1i, r2i, r3i, r4i] = [
    tf.zeros([1, 1, 1, 1]),
    tf.zeros([1, 1, 1, 1]),
    tf.zeros([1, 1, 1, 1]),
    tf.zeros([1, 1, 1, 1]),
  ];
  const downsample_ratio = tf.scalar(0.5);

  // Run model
  const [fgr, pha] = (await model.executeAsync(
    { src, r1i, r2i, r3i, r4i, downsample_ratio },
    ["fgr", "pha"]
  )) as [Tensor<Rank.R2>, Tensor<Rank.R3>];

  // Prepare background tensor (resized to webcam frame)
  const bgTensor = tf.tidy(() => {
    const bg = tf.browser
      .fromPixels(backgroundImg)
      .resizeBilinear([480, 640])
      .div(255);
    return bg.expandDims(0);
  });

  // ✅ Enhance mask (pha)
  const phaUpscaled = tf.tidy(() => {
    // 1. Upscale to input resolution
    let up = tf.image.resizeBilinear(pha, [480, 640], true);
    // 2. Smooth with average pooling (acts like blur)
    up = tf.avgPool(up, [5, 5], [1, 1], "same");
    // 3. Keep it in range [0,1]
    return tf.clipByValue(up, 0, 1);
  });

  // Composite foreground with new background
  const composited = tf.tidy(() => {
    const pha3 = phaUpscaled.tile([1, 1, 1, 3]); // repeat alpha for RGB
    return fgr
      .resizeBilinear([480, 640])
      .mul(pha3)
      .add(bgTensor.mul(tf.sub(1, pha3)))
      .squeeze();
  });

  await browser.toPixels(composited as Tensor<Rank.R2>, canvas);

  // Cleanup
  tf.dispose([img, src, fgr, pha, bgTensor, composited, phaUpscaled]);
}
