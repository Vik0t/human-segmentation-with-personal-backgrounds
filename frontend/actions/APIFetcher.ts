"use server";

import { BGData } from "@/types/BGData";

export async function getBGs(uid: number): Promise<BGData[] | null> {
  const data = [
    { id: 1, img: "bg1.png" },
    { id: 2, img: "bg2.png" },
    { id: 3, img: "bg3.png" },
    { id: 4, img: "bg4.png" },
  ]; // setting mock data instead of fetching
  const result = BGData.array().safeParse(data);
  if (!result.success) {
    return null;
  }
  return result.data;
}
