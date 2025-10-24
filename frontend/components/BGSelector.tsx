"use client"
import { BGData } from "@/types/BGData"
import { Dialog } from "@base-ui-components/react"
import { useEffect, useState, useTransition } from "react"
import { getBGs, setBG } from "@/actions/APIFetcher"

export default function BGSelector({uid, children, className}:{uid:number, children: React.ReactNode, className:string}) {
  const [selectedBG, setSelectedBG] = useState<BGData | null>(null)
  const [isPending, startTransition] = useTransition()
  useEffect(() => {
    if (!selectedBG) return
    startTransition(async () => {
      await setBG(uid, selectedBG.id)
    })
  }, [selectedBG])

  return (
    <Dialog.Root>
      <Dialog.Trigger className={className}>
        {children}
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Backdrop className="fixed inset-0 min-h-dvh bg-black opacity-20 transition-all duration-150 data-ending-style:opacity-0 data-starting-style:opacity-0 dark:opacity-70 supports-[-webkit-touch-callout:none]:absolute" />
        <Dialog.Popup className="fixed top-1/2 left-1/2 -mt-8 w-96 max-w-[calc(100vw-3rem)] -translate-x-1/2 -translate-y-1/2 rounded-lg bg-gray-50 p-6 text-gray-900 outline-1 outline-gray-200 transition-all duration-150 data-ending-style:scale-90 data-ending-style:opacity-0 data-starting-style:scale-90 data-starting-style:opacity-0 dark:outline-gray-300">
          <BGList uid={uid} selectedBG={selectedBG} setSelectedBG={setSelectedBG} />
          {isPending && <div className="w-full h-full bg-white opacity-20 absolute top-0 left-0 z-10 animate-pulse"></div>}
        </Dialog.Popup>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
function BGList({
  uid,
  selectedBG,
  setSelectedBG,
}: {
  uid: number
  selectedBG: BGData | null
  setSelectedBG: (bg: BGData) => void
}) {
  const [isPending, startTransition] = useTransition()
  const [bgs, setBgs] = useState<BGData[]>([])
  const refresh = () => {
    startTransition(async () => {
      const bgs = await getBGs(uid)
      if (!bgs) return
      setBgs(bgs)
    })
  }
  useEffect(() => {
    refresh()
  }, [uid])
  return (
    <>
      <div className="flex flex-row justify-between items-center">
        <Dialog.Title className="-mt-1.5 mb-1 text-lg font-medium">Backgrounds</Dialog.Title>
        <button
          className="text-sm text-gray-500"
          disabled={isPending}
          onClick={() => {
            refresh()
          }}
        >
          Refresh
        </button>
      </div>
      <div className="flex flex-row gap-2 flex-wrap">
        {isPending ? (
          <div className="w-24 h-24 rounded-md overflow-hidden border border-gray-200 animate-pulse bg-gray-200">
            <div className="w-full h-full animate-pulse bg-gray-200"></div>
          </div>
        ) : (
          bgs.map((bg) => (
            <BGPreview
              key={bg.id}
              bg={bg}
              onClick={() => {
                setSelectedBG(bg)
              }}
              selected={selectedBG?.id === bg.id}
            />
          ))
        )}
      </div>
      <div className="flex justify-end gap-4">
        <Dialog.Close className="flex h-10 items-center justify-center rounded-md border border-gray-200 bg-gray-50 px-3.5 text-base font-medium text-gray-900 select-none hover:bg-gray-100 focus-visible:outline-2 focus-visible:-outline-offset-1 focus-visible:outline-blue-800 active:bg-gray-100">
          Close
        </Dialog.Close>
      </div>
    </>
  )
}

function BGPreview({ bg, onClick, selected }: { bg: BGData; onClick: () => void; selected: boolean }) {
  return (
    <div
      key={bg.id}
      className={`w-24 h-24 rounded-md overflow-hidden border p-1 ${selected ? "border-blue-500" : "border-gray-200"}`}
      onClick={onClick}
    >
      <img src={`data:image/jpeg;base64,${bg.img}`} alt={bg.id.toString()} className="w-full h-full object-cover" />
    </div>
  )
}
