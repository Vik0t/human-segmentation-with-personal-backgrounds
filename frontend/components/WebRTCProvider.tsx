"use client"
import { createContext, ReactNode, useContext, useEffect, useRef, useState } from "react"
import { start } from "@/actions/APIFetcher"

export const WebRTCProviderContext = createContext<WebRTCContextType | null>(null)
interface WebRTCContextType {
  videoStreams: readonly MediaStream[]
  isConnected: boolean
  openOffer: () => Promise<void>
  closeOffer: () => Promise<void>
  registerErrorHandler: (handler: (error: Error) => void) => void
  unregisterErrorHandler: (handler: (error: Error) => void) => void
}

export const useWebRTC = () => {
  const context = useContext(WebRTCProviderContext)
  if (!context) {
    throw new Error("useWebRTC must be used within a WebRTCProvider")
  }
  return context
}

export default function WebRTCProvider({ children }: { children: ReactNode }) {
  const [isConnected, setIsConnected] = useState(false)
  const [videoStreams, setVideoStreams] = useState<readonly MediaStream[]>([])
  const pcRef = useRef<RTCPeerConnection | null>(null)
  const isIceCompleteRef = useRef(false)
  const errorHandlersRef = useRef<((error: Error) => void)[]>([])

  const registerErrorHandler = (handler: (error: Error) => void) => {
    errorHandlersRef.current.push(handler)
  }
  const unregisterErrorHandler = (handler: (error: Error) => void) => {
    errorHandlersRef.current = errorHandlersRef.current.filter((h) => h !== handler)
  }
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
    pcRef.current.addTransceiver("video", { direction: "recvonly" })
    pcRef.current.addEventListener("track", (evt) => {
      if (evt.track.kind === "video") {
        setVideoStreams(evt.streams)
      }
    })
    pcRef.current.addEventListener("icegatheringstatechange", () => {
      if (pcRef.current!.iceGatheringState === "complete") {
        isIceCompleteRef.current = true
      }
    })
  }, [isConnected])

  const openOffer = async () => {
    console.log("Creating offer")
    const offerData = await pcRef.current!.createOffer()
    await pcRef.current!.setLocalDescription(offerData)
    console.log("Setting local description")
    await new Promise<void>((resolve, reject) => {
      if (pcRef.current!.iceGatheringState === "complete") {
        resolve()
      } else {
        setTimeout(() => {
          if (pcRef.current!.iceGatheringState !== "complete") {
            pcRef.current!.removeEventListener("icegatheringstatechange", checkState)
            reject(new Error("ICE gathering failed"))
          }
        }, 10000)
        const checkState = () => {
          if (pcRef.current!.iceGatheringState === "complete") {
            pcRef.current!.removeEventListener("icegatheringstatechange", checkState)
            resolve()
          }
        }
        pcRef.current!.addEventListener("icegatheringstatechange", checkState)
      }
    })
    console.log("ICE gathering complete")
    if (!isIceCompleteRef.current) {
      const error = new Error("ICE gathering failed")
      errorHandlersRef.current.forEach((h) => h(error))
      console.error(error)
      return
    }
    const offer = pcRef.current!.localDescription
    if (!offer) {
      const error = new Error("Offer is null")
      errorHandlersRef.current.forEach((h) => h(error))
      console.error(error)
      return
    }
    const answerData = await start(1, offer.sdp, offer.type)
    if (!answerData) {
      const error = new Error("Failed to start videostream")
      errorHandlersRef.current.forEach((h) => h(error))
      console.error(error)
      return
    }
    await pcRef.current!.setRemoteDescription(answerData)
    setIsConnected(true)
  }

  const closeOffer = async () => {
    await pcRef.current!.close()
    setIsConnected(false)
    setVideoStreams([])
  }

  return (
    <WebRTCProviderContext
      value={{
        videoStreams: videoStreams,
        isConnected: isConnected,
        openOffer: openOffer,
        closeOffer: closeOffer,
        registerErrorHandler: registerErrorHandler,
        unregisterErrorHandler: unregisterErrorHandler,
      }}
    >
      {children}
    </WebRTCProviderContext>
  )
}
