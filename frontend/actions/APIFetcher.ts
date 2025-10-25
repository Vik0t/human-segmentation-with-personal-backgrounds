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

export async function getUniqueBG(
  srcImg: string,
  employeeData: {[key: string]: string}
): Promise<Blob | null> {

        const payload = {
            employee: employeeData,
            background_base64: srcImg
        }
        console.log(JSON.stringify(payload))
  try {
    const result = await fetch("http://localhost:8000/generate_background", {
      method: "POST",
      headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
      
    });
    if (!result.ok) {
      return null;
    }
    const data = await result.blob();
    return data;
  } catch (error) {
    console.error("Error fetching unique background:", error);
    return null;
  }
}
