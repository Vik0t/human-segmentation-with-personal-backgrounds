"use client"
import {
  GraphModel,
  loadGraphModel,
  nextFrame,
  data as tfdata,
  dispose as tfdispose,
  tidy as tftidy,
  tensor as tftensor,
  fill as tfFill,
  concat as tfConcat,
  Tensor,
  Rank,
} from "@tensorflow/tfjs"
import { createContext, ReactNode, RefObject, useContext, useRef, useState } from "react"

interface TFSegmentationContextType {
  srcVideoref: RefObject<HTMLVideoElement | null>
  canvasref: RefObject<HTMLCanvasElement | null>
  start: () => Promise<void>
  stop: () => void
  setBG: (bg: string) => void
  setJSONConfig: (jsonConfig: string) => void
}
const TFSegmentationProviderContext = createContext<TFSegmentationContextType | null>(null)

export function useTFSegmentation() {
  const context = useContext(TFSegmentationProviderContext)
  if (!context) {
    throw new Error("useTFSegmentation must be used within a TFSegmentationProvider")
  }
  return context
}

export default function TFSegmentationProvider({ children }: { children: ReactNode }) {
  const srcVideoref = useRef<HTMLVideoElement>(null)
  const canvasref = useRef<HTMLCanvasElement>(null)
  const [isStarted, setIsStarted] = useState(false)
  const [bg, setBG] = useState<string | null>(null)
  const [jsonConfig, setJSONConfig] = useState<string | null>(null)
  const modelref = useRef<GraphModel | null>(null)
  const webcamref = useRef<any>(null)
  const nextFrameref = useRef<number>(0)
  const tensorsRef = useRef<{ r11: Tensor; r21: Tensor; r31: Tensor; r41: Tensor } | null>(null)

  const draw = async () => {
    console.log("drawing")
    if (!modelref.current || !webcamref.current || !canvasref.current || !tensorsRef.current) return
    console.log("modelref", modelref.current)
    await nextFrame()
    console.log("nextFrame")
    const img = await webcamref.current.capture()
    console.log("img", img.shape)
    const src = tftidy(() => img.expandDims(0).div(255)) // normalize input
    console.log("src", src.shape)
    const [fgr, pha, r1o, r2o, r3o, r4o] = (await modelref.current.executeAsync(
      {
        src,
        ...tensorsRef.current,
        downsample_ratio: tftensor(0.5),
      }, // provide inputs as a single object, not array
      ["fgr", "pha", "r1o", "r2o", "r3o", "r4o"] // select outputs
    )) as [Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
    // Draw the result based on selected view
    console.log("executed")
    if (bg === "white") {
      drawMatte(fgr.clone(), pha.clone())
      canvasref.current.style.background = "rgb(255, 255, 255)"
    } else if (bg === "green") {
      drawMatte(fgr.clone(), pha.clone())
      canvasref.current.style.background = "rgb(120, 255, 155)"
    } else if (bg === "alpha") {
      drawMatte(null, pha.clone())
      canvasref.current.style.background = "rgb(0, 0, 0)"
    } else if (bg === "foreground") {
      drawMatte(fgr.clone(), null)
    }
    console.log("drawn")

    // Dispose old tensors.
    tfdispose([img, src, fgr, pha, r1o, r2o, r3o, r4o])
    tensorsRef.current = { r11: r1o, r21: r2o, r31: r3o, r41: r4o }

    // Update recurrent states.
    nextFrameref.current = requestAnimationFrame(draw)
  }

  async function drawMatte(fgr: any, pha: any) {
    if (!canvasref.current) return
    const rgba = tftidy(() => {
      const rgb =
        fgr !== null ? fgr.squeeze(0).mul(255).cast("int32") : tfFill([pha.shape[1], pha.shape[2], 3], 255, "int32")
      const a =
        pha !== null ? pha.squeeze(0).mul(255).cast("int32") : tfFill([fgr.shape[1], fgr.shape[2], 1], 255, "int32")
      return tfConcat([rgb, a], -1)
    })
    fgr && tfdispose(fgr)
    pha && tfdispose(pha)
    const [height, width] = rgba.shape.slice(0, 2)
    const pixelData = new Uint8ClampedArray(await rgba.data())
    const imageData = new ImageData(pixelData, width, height)
    canvasref.current.width = width
    canvasref.current.height = height
    canvasref.current.getContext("2d")!.putImageData(imageData, 0, 0)
    rgba.dispose()
  }

  const start = async () => {
    if (isStarted || !srcVideoref.current || !canvasref.current) return
    setIsStarted(true)
    console.log("starting tf segmentation")
    modelref.current = await loadGraphModel("model.json")
    console.log("model loaded")
    srcVideoref.current.width = 640
    srcVideoref.current.height = 480
    webcamref.current = await tfdata.webcam(srcVideoref.current)
    console.log("webcam started")
    tensorsRef.current = { r11: tftensor(0), r21: tftensor(0), r31: tftensor(0), r41: tftensor(0) }
    draw()
  }
  const stop = () => {
    if (!isStarted) return
    setIsStarted(false)
    cancelAnimationFrame(nextFrameref.current)
    webcamref.current.stop()
    webcamref.current = null
    modelref.current = null
    tensorsRef.current = null
    nextFrameref.current = 0
  }
  return (
    <TFSegmentationProviderContext value={{ srcVideoref, canvasref, start, stop, setBG, setJSONConfig }}>
      {children}
    </TFSegmentationProviderContext>
  )
}
