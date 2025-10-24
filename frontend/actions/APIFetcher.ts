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
    return false
  }
  const data = await response.json()
  const result = BGData.array().safeParse(data)
  if (!result.success) {
    return false
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
    return false
  }
  if (!response.ok) {
    return false
  }
  return true
}

export async function start(uid: number, sdp: string, type: string) {
  try{
  const response = await fetch(`${process.env.PROCESSING_API_IP}/offer`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ uid, sdp, type }),
  })
  if (!response.ok) {
    return null
    }
    return await response.json()
  } catch (error) {
    return null
  }
}

export async function SendjsonConfig(uid: number, config: string) {
  const response = await fetch(`${process.env.PROCESSING_API_IP}/config`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ uid, config }),
  })
  if (!response.ok) {
    return false
  }
  return true
}