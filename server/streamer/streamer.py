from aiortc import (
    VideoStreamTrack,
    RTCPeerConnection,
    RTCSessionDescription,
)
from processorABC import BGData, ProcessorABC
from av.video.frame import VideoFrame
from typing import Type
from aiohttp import web
import numpy as np
import base64
import json
import cv2


class VideoStreamTrackWithProcessor(VideoStreamTrack):
    __processor: ProcessorABC
    __cap: cv2.VideoCapture

    def __init__(self, prc: ProcessorABC) -> None:
        super().__init__()
        self.__processor = prc
        self.__cap = cv2.VideoCapture(0)

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        if self.__cap.isOpened():
            ret, frame = self.__cap.read()
            if not ret:
                raise Exception("Error: Could not read frame from camera")
        else:
            raise Exception("Error: Could not open camera")
        prc_frame = self.__processor.process_image(frame)
        video_frame = VideoFrame.from_ndarray(
            prc_frame, format="bgr24"  # pyright: ignore[reportArgumentType]
        )
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


class Streamer:
    __pcs: dict[int, RTCPeerConnection]
    __processors: dict[int, ProcessorABC]
    __Procssor: Type[ProcessorABC]
    __app: web.Application

    def __init__(self, Processor: Type[ProcessorABC]) -> None:
        self.__Procssor = Processor
        self.__processors = {}
        self.__pcs = {}
        self.__app = web.Application()
        self.__app.router.add_post("/available_bgs", lambda r: self.get_bgs(r))
        self.__app.router.add_post("/bgs", lambda r: self.set_bg(r))
        self.__app.router.add_post("/offer", lambda r: self.process_offer(r))
        self.__app.router.add_post("/config", lambda r: self.receive_config(r))
        web.run_app(self.__app, port=5000)

    def __encode_img(self, img: cv2.typing.MatLike) -> str:
        _, buffer = cv2.imencode(".jpg", img)
        return base64.b64encode(buffer).decode("utf-8")

    async def get_bgs(self, r: web.Request) -> web.Response:
        print("Getting bgs")
        try:
            uid = int((await r.json())["uid"])
        except KeyError:
            return web.HTTPBadRequest(reason="Missing 'uid' query parameters")
        if uid not in self.__processors:
            self.__processors[uid] = self.__Procssor(uid)
        bg_list = self.__processors[uid].get_bg_list()
        return web.json_response(
            [{"id": bg.id, "img": self.__encode_img(bg.img)} for bg in bg_list]
        )

    async def set_bg(self, r: web.Request) -> web.Response:
        print("Setting bg")
        query = await r.json()
        try:
            id = int(query["bgid"])
        except KeyError:
            return web.HTTPBadRequest(reason="Missing 'bgid' query parameters")
        try:
            uid = int(query["uid"])
        except KeyError:
            return web.HTTPBadRequest(reason="Missing 'uid' query parameters")
        if uid not in self.__processors:
            self.__processors[uid] = self.__Procssor(uid)
        success = self.__processors[uid].set_bg(id)
        if not success:
            return web.HTTPBadRequest(reason="Failed to set background")
        return web.Response(content_type="text/plain", text="OK")
    
    async def receive_config(self, r: web.Request) -> web.Response:
        print("Receiving config")
        query = await r.json()
        try:
            uid = int(query["uid"])
        except KeyError:
            return web.HTTPBadRequest(reason="Missing 'uid' query parameters")
        if uid not in self.__processors:
            self.__processors[uid] = self.__Procssor(uid)
        success = self.__processors[uid].receive_config(query["config"])
        if not success:
            return web.HTTPBadRequest(reason="Failed to receive config")
        return web.Response(content_type="text/plain", text="OK")

    async def process_offer(self, r: web.Request):
        print("Processing offer")
        params = await r.json()
        try:
            offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        except KeyError:
            return web.HTTPBadRequest(reason="Malformed offer")
        try:
            uid = params["uid"]
        except KeyError:
            return web.HTTPBadRequest(reason="Missing 'uid' query parameters")
        if uid not in self.__processors:
            self.__processors[uid] = self.__Procssor(uid)
        video = VideoStreamTrackWithProcessor(self.__processors[uid])

        pc = RTCPeerConnection()
        self.__pcs[uid] = pc

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            print("Connection state is %s" % pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                self.__pcs.pop(uid)
                self.__processors.pop(uid)

        # open media source
        pc.addTrack(video)

        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        print("Local description set, sending offer")

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        )


# Example
if __name__ == "__main__":

    class EmptyProcessor(ProcessorABC):
        __bg_id: int

        def __init__(self, uid: int):
            super().__init__(uid)
            self.__bg_id = 0

        def process_image(self, img: cv2.typing.MatLike) -> cv2.typing.MatLike:
            # return cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
            if self.__bg_id == 1:
                return cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1]
            elif self.__bg_id == 2:
                return cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
            else:
                return img

        def set_bg(self, bg_id: int) -> bool:
            if not (bg_id == 1 or bg_id == 2):
                return False
            self.__bg_id = bg_id
            return True

        def receive_config(self, config: dict) -> bool:
            return True

        def get_bg_list(self) -> list[BGData]:
            img1 = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
            img2 = np.random.randint(0, 255, size=(50, 50, 3), dtype=np.uint8)
            return [BGData(1, img1), BGData(2, img2)]

    streamer = Streamer(EmptyProcessor)
