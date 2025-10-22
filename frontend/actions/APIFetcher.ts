"use server"

import { BGData } from "@/types/BGData"

export async function getBGs(uid: number) {
  const response = await fetch(`${process.env.PROCESSING_API_IP}/available_bgs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ uid }),
  })
  if (!response.ok) {
    throw new Error("Failed to fetch BGs")
  }
  const data = await response.json()
  const result = BGData.array().safeParse(data)
  if (!result.success) {
    throw new Error("Failed to parse BGs")
  }
  return result.data
}

export async function setBG(uid: number, bgid: number) {
  const response = await fetch(`${process.env.PROCESSING_API_IP}/bgs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ uid, bgid }),
  })
  if (!response.ok) {
    throw new Error("Failed to fetch BG")
  }
  if (!response.ok) {
    throw new Error("Failed to set BG")
  }
  return true
}

export async function start(uid: number, sdp: string, type: string) {
  const response = await fetch(`${process.env.PROCESSING_API_IP}/offer`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ uid, sdp, type }),
  })
  if (!response.ok) {
    throw new Error("Failed to start")
  }
  return await response.json()
}