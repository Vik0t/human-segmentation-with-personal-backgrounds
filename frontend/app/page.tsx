"use client"

import { start } from "@/actions/APIFetcher"
import BGSelector from "@/components/BGSelector"
import { useRef, useState, useTransition, useEffect } from "react"

export default function Home() {
  const [isConnected, setIsConnected] = useState(false)
  const [isPending, startTransition] = useTransition()

  const srcVideoRef = useRef<HTMLVideoElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const pcRef = useRef<RTCPeerConnection | null>(null)
  const isIceCompleteRef = useRef(false)

  const checkState = () => {
    if (pcRef.current!.iceGatheringState === "complete") {
      pcRef.current!.removeEventListener("icegatheringstatechange", checkState)
      isIceCompleteRef.current = true
    }
  }

  useEffect(() => {
    if (isConnected) return
    const config = {
      sdpSemantics: "unified-plan",
      iceServers: [{ urls: ["stun:stun.l.google.com:19302"] }],
    }
    pcRef.current = new RTCPeerConnection(config)
    pcRef.current.addEventListener("track", (evt) => {
      if (evt.track.kind === "video") {
        videoRef.current!.srcObject = evt.streams[0]
      }
    })
    pcRef.current.addEventListener("icegatheringstatechange", () => {
      if (pcRef.current!.iceGatheringState === "complete") {
        isIceCompleteRef.current = true
      }
    })
  }, [isConnected])

  return (
    <div className="h-full w-full flex flex-col items-center justify-center gap-4 bg-gray-100 p-8">
      <h1 className="m-0 text-3xl font-bold text-center text-gray-800">WebRTC Webcam</h1>
      <div className="h-full bg-gray-200 rounded-lg overflow-hidden relative">
        <video
          ref={srcVideoRef}
          autoPlay
          playsInline
          className={`absolute top-5 left-5 w-30% h-30% object-cover border-2 border-gray-300 rounded-lg ${isConnected ? "block" : "hidden"}`}
        />
        {!isConnected && <div className="absolute top-0 left-0 w-full h-full bg-gray-500 opacity-50 text-white text-center flex items-center justify-center text-5xl font-bold">Press 'start' to start the stream</div>}
        <video ref={videoRef} autoPlay playsInline className="w-full h-full object-cover" />
      </div>
      <div className="flex flex-row gap-4 justify-center h-10">
        <BGSelector uid={1} />
        <button
          onClick={() => {
            if (isConnected) {
              pcRef.current!.close()
              videoRef.current!.srcObject = null
              srcVideoRef.current!.srcObject = null
              setIsConnected(false)
              return
            }
            startTransition(async () => {
              const localStream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: false,
              });
              srcVideoRef.current!.srcObject = localStream
              localStream.getTracks().forEach((track) => {
                pcRef.current!.addTrack(track)
              })
              const offerData = await pcRef.current!.createOffer()
              await pcRef.current!.setLocalDescription(offerData)
              await new Promise<void>((resolve) => {
                if (pcRef.current!.iceGatheringState === "complete") {
                  resolve()
                } else {
                  const checkState = () => {
                    if (pcRef.current!.iceGatheringState === "complete") {
                      pcRef.current!.removeEventListener("icegatheringstatechange", checkState)
                      resolve()
                    }
                  }
                  pcRef.current!.addEventListener("icegatheringstatechange", checkState)
                }
              })
              if (!isIceCompleteRef.current) {
                throw new Error("ICE gathering failed")
              }
              const offer = pcRef.current!.localDescription
              if (!offer) {
                throw new Error("Offer is null")
              }
              const answerData = await start(1, offer.sdp, offer.type)
              await pcRef.current!.setRemoteDescription(answerData)
              setIsConnected(true)
            })
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
