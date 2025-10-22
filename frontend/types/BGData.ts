import { z } from "zod"

export const BGData = z.object({
  id: z.number(),
  img: z.string(),
})

export type BGData = z.infer<typeof BGData>