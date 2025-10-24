"use client"
import TFSegmentationProvider, { useTFSegmentation } from "@/components/TFSegmentationProvider";
import { useTransition } from "react";

export default function TmpTest() {
  return <TFSegmentationProvider>
    <TestContent />
  </TFSegmentationProvider>
}

function TestContent() {
  const [isPending, startTransition] = useTransition()
  const { start, stop, setBG, srcVideoref, canvasref } = useTFSegmentation()
  return <div>
    {isPending && <div className="w-full h-full bg-white opacity-20 absolute top-0 left-0 z-10 animate-pulse"></div>}
    <div className="flex flex-col gap-6 items-center mt-10">
      <div className="flex flex-wrap gap-3 mb-4">
        <button
          className="px-4 py-2 rounded bg-blue-600 text-white font-semibold hover:bg-blue-700 shadow"
          onClick={() => startTransition(start)}
        >
          Start
        </button>
        <button
          className="px-4 py-2 rounded bg-gray-300 text-gray-900 font-semibold hover:bg-gray-400 shadow"
          onClick={stop}
        >
          Stop
        </button>
        <button
          className="px-4 py-2 rounded bg-white ring-1 ring-gray-300 text-gray-800 font-semibold hover:bg-gray-50 shadow"
          onClick={() => setBG("white")}
        >
          White
        </button>
        <button
          className="px-4 py-2 rounded bg-green-400 text-white font-semibold hover:bg-green-500 shadow"
          onClick={() => setBG("green")}
        >
          Green
        </button>
        <button
          className="px-4 py-2 rounded bg-black text-white font-semibold hover:bg-gray-800 shadow"
          onClick={() => setBG("alpha")}
        >
          Alpha
        </button>
        <button
          className="px-4 py-2 rounded bg-purple-600 text-white font-semibold hover:bg-purple-700 shadow"
          onClick={() => setBG("foreground")}
        >
          Foreground
        </button>
      </div>
      <div className="relative w-[640px] mx-auto border rounded-lg shadow bg-gray-100 overflow-hidden">
        <video ref={srcVideoref} />
        <canvas ref={canvasref} />
      </div>
    </div>
  </div>
}