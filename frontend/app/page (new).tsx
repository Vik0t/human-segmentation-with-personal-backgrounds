"use client"

import BGSelector from "@/components/BGSelector"
import WebRTCProvider, { useWebRTC } from "@/components/WebRTCProvider"
import { Toast } from "@base-ui-components/react"
import { useEffect, useRef, useTransition } from "react"

export default function Home() {
  return (
    <WebRTCProvider>
      <Toast.Provider>
        <HomeContent />
        <Toast.Portal>
          <Toast.Viewport className="fixed z-10 top-auto right-4 bottom-4 mx-auto flex w-[250px] sm:right-8 sm:bottom-8 sm:w-[300px]">
            <ToastList />
          </Toast.Viewport>
        </Toast.Portal>
      </Toast.Provider>
    </WebRTCProvider>
  )
}

function HomeContent() {
  const [isPending, startTransition] = useTransition()
  const { videoStreams, isConnected, openOffer, closeOffer, registerErrorHandler, unregisterErrorHandler } = useWebRTC()
  const toast = Toast.useToastManager()
  const videoRef = useRef<HTMLVideoElement>(null)
  useEffect(() => {
    if (videoStreams.length > 0) {
      videoRef.current!.srcObject = videoStreams[0]
    }
  }, [videoStreams])
  const errorHandler = (error: Error) => {
    toast.add({
      title: "Error occured while starting videostream",
      description: error.message,
    })
  }
  useEffect(() => {
    registerErrorHandler(errorHandler)
    return () => {
      unregisterErrorHandler(errorHandler)
    }
  }, [])

  return (
    <div className="h-full w-full flex flex-col items-center justify-center gap-4 bg-gray-100 p-8">
      <h1 className="m-0 text-3xl font-bold text-center text-gray-800">WebRTC Webcam</h1>
      <div className="h-full bg-gray-200 rounded-lg overflow-hidden relative">
        {!isConnected && (
          <div className="absolute top-0 left-0 w-full h-full bg-gray-500 opacity-50 text-white text-center flex items-center justify-center text-5xl font-bold">
            Press 'start' to start the stream
          </div>
        )}
        <video ref={videoRef} autoPlay playsInline className="w-full h-full object-cover" />
      </div>
      <div className="flex flex-row gap-4 justify-center h-10">
        <BGSelector uid={1} className="flex flex-col items-center justify-center w-full h-10 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200 font-medium">
          <div className="flex flex-col items-center justify-center w-full h-10 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200 font-medium">
            Select Background
          </div>
        </BGSelector>
        <button
          onClick={() => {
            if (isConnected) {
              closeOffer()
              return
            }
            startTransition(openOffer)
          }}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200 font-medium"
          disabled={isPending}
        >
          {isConnected ? "Stop" : "Start"}
          {isPending && <span className="ml-2 animate-pulse">...</span>}
        </button>
      </div>
    </div>
  )
}

function ToastList() {
  const { toasts } = Toast.useToastManager()
  return toasts.map((toast) => (
    <Toast.Root
      key={toast.id}
      toast={toast}
      className="[--gap:0.75rem] [--peek:0.75rem] [--scale:calc(max(0,1-(var(--toast-index)*0.1)))] [--shrink:calc(1-var(--scale))] [--height:var(--toast-frontmost-height,var(--toast-height))] [--offset-y:calc(var(--toast-offset-y)*-1+calc(var(--toast-index)*var(--gap)*-1)+var(--toast-swipe-movement-y))] absolute right-0 bottom-0 left-auto z-[calc(1000-var(--toast-index))] mr-0 w-full origin-bottom transform-[translateX(var(--toast-swipe-movement-x))_translateY(calc(var(--toast-swipe-movement-y)-(var(--toast-index)*var(--peek))-(var(--shrink)*var(--height))))_scale(var(--scale))] rounded-lg border border-gray-200 bg-gray-50 bg-clip-padding p-4 shadow-lg select-none after:absolute after:top-full after:left-0 after:h-[calc(var(--gap)+1px)] after:w-full after:content-[''] data-ending-style:opacity-0 data-expanded:transform-[translateX(var(--toast-swipe-movement-x))_translateY(calc(var(--offset-y)))] data-limited:opacity-0 data-starting-style:transform-[translateY(150%)] [&[data-ending-style]:not([data-limited]):not([data-swipe-direction])]:transform-[translateY(150%)] data-ending-style:data-[swipe-direction=down]:transform-[translateY(calc(var(--toast-swipe-movement-y)+150%))] data-expanded:data-ending-style:data-[swipe-direction=down]:transform-[translateY(calc(var(--toast-swipe-movement-y)+150%))] data-ending-style:data-[swipe-direction=left]:transform-[translateX(calc(var(--toast-swipe-movement-x)-150%))_translateY(var(--offset-y))] data-expanded:data-ending-style:data-[swipe-direction=left]:transform-[translateX(calc(var(--toast-swipe-movement-x)-150%))_translateY(var(--offset-y))] data-ending-style:data-[swipe-direction=right]:transform-[translateX(calc(var(--toast-swipe-movement-x)+150%))_translateY(var(--offset-y))] data-expanded:data-ending-style:data-[swipe-direction=right]:transform-[translateX(calc(var(--toast-swipe-movement-x)+150%))_translateY(var(--offset-y))] data-ending-style:data-[swipe-direction=up]:transform-[translateY(calc(var(--toast-swipe-movement-y)-150%))] data-expanded:data-ending-style:data-[swipe-direction=up]:transform-[translateY(calc(var(--toast-swipe-movement-y)-150%))] h-(--height) data-expanded:h-(--toast-height) [transition:transform_0.5s_cubic-bezier(0.22,1,0.36,1),opacity_0.5s,height_0.15s]"
    >
      <Toast.Content className="overflow-hidden transition-opacity [transition-duration:250ms] data-behind:pointer-events-none data-behind:opacity-0 data-expanded:pointer-events-auto data-expanded:opacity-100">
        <Toast.Title className="text-[0.975rem] leading-5 font-medium" />
        <Toast.Description className="text-[0.925rem] leading-5 text-gray-700" />
        <Toast.Close
          className="absolute top-2 right-2 flex h-5 w-5 items-center justify-center rounded border-none bg-transparent text-gray-500 hover:bg-gray-100 hover:text-gray-700"
          aria-label="Close"
        >
          <XIcon className="h-4 w-4" />
        </Toast.Close>
      </Toast.Content>
    </Toast.Root>
  ))
}

function XIcon(props: React.ComponentProps<"svg">) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      <path d="M18 6 6 18" />
      <path d="m6 6 12 12" />
    </svg>
  )
}
