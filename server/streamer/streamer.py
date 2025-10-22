from aiortc import VideoStreamTrack, RTCPeerConnection, RTCRtpSender, RTCSessionDescription
from processorABC import BGData, ProcessorABC
from av.video.frame import VideoFrame
from typing import Optional, Type
from aiohttp import web
import numpy as np
import asyncio
import base64
import json
import cv2


class VideoStreamTrackWithProcessor(VideoStreamTrack):
    __processor: ProcessorABC
    __track: Optional[VideoStreamTrack]

    def __init__(self, prc: ProcessorABC) -> None:
        super().__init__()
        self.__processor = prc
        self.__track = None
    
    def set_track(self, track:VideoStreamTrack):
        self.__track = track
    
    async def recv(self):
        pts, time_base = await self.next_timestamp()
        if self.__track is None:
            print("No input track available, waiting...")
            while self.__track is None:
                await asyncio.sleep(0.1)
            print("Track received")

        frame = await asyncio.wait_for(self.__track.recv(), timeout=5.0)
        if isinstance(frame, VideoFrame):
            frame = frame.to_ndarray(format="bgr24")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # pyright: ignore[reportArgumentType, reportCallIssue]
        prc_frame = self.__processor.process_image(frame)

        video_frame = VideoFrame.from_ndarray(prc_frame, format="rgb24")  # pyright: ignore[reportArgumentType]
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


class Streamer:
    __pcs: dict[int, RTCPeerConnection]
    __processors: dict[int, ProcessorABC]
    __Procssor: Type[ProcessorABC]
    __app: web.Application

    def __init__(self, Processor: Type[ProcessorABC], debug=False) -> None:
        self.__Procssor = Processor
        self.__processors = {}
        self.__pcs = {}
        self.__app = web.Application()
        if debug: self.__app.router.add_get("/", lambda r: self.index())
        self.__app.router.add_get("/available_bgs", lambda r: self.get_bgs(r))
        self.__app.router.add_get("/bgs/{id}", lambda r: self.get_bg(r))
        self.__app.router.add_post("/offer", lambda r: self.process_offer(r))
        web.run_app(self.__app, port=5000)

    def __encode_img(self, img: cv2.typing.MatLike) -> str:
        _, buffer = cv2.imencode(".jpg", img)
        return base64.b64encode(buffer).decode('utf-8')
    
    async def index(self) -> web.Response:
        return web.Response(content_type="text/html", text="""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <title>WebRTC webcam</title>
            <style>
            button {
                padding: 8px 16px;
            }

            video {
                width: 100%;
            }

            .option {
                margin-bottom: 8px;
            }

            #media {
                max-width: 1280px;
            }
            .video-container {
                display: flex;
                flex-direction: row;
                width: 100vw;
                height: 60vh;
            }
            </style>
        </head>
        <body>
            <div class="option">
            <input id="use-stun" type="checkbox" />
            <label for="use-stun">Use STUN server</label>
            </div>
            <button id="start" onclick="start()">Start</button>
            <button id="stop" style="display: none" onclick="stop()">Stop</button>

            <div id="media">
            <h2>Media</h2>
            <div class="video-container">
                <video id="srcvideo" autoplay="true" playsinline="true"></video>
                <video id="video" autoplay="true" playsinline="true"></video>
            </div>
            </div>

            <script>
            var pc = null;

        async function negotiate() {
        pc.addTransceiver("video", { direction: "sendrecv" });
        const localStream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false,
        });
        localStream.getTracks().forEach((track) => {
            pc.addTrack(track, localStream);
        });
        document.getElementById("srcvideo").srcObject = localStream;
        return pc
            .createOffer()
            .then((offer) => {
            return pc.setLocalDescription(offer);
            })
            .then(() => {
            return new Promise((resolve) => {
                if (pc.iceGatheringState === "complete") {
                resolve();
                } else {
                const checkState = () => {
                    if (pc.iceGatheringState === "complete") {
                    pc.removeEventListener("icegatheringstatechange", checkState);
                    resolve();
                    }
                };
                pc.addEventListener("icegatheringstatechange", checkState);
                }
            });
            })
            .then(() => {
            var offer = pc.localDescription;
            return fetch("/offer", {
                body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
                uid: 1,
                }),
                headers: {
                "Content-Type": "application/json",
                },
                method: "POST",
            });
            })
            .then((response) => {
            return response.json();
            })
            .then((answer) => {
            return pc.setRemoteDescription(answer);
            })
            .catch((e) => {
            alert(e);
            });
        }

        function start() {
        var config = {
            sdpSemantics: "unified-plan",
        };

        if (document.getElementById("use-stun").checked) {
            config.iceServers = [{ urls: ["stun:stun.l.google.com:19302"] }];
        }

        pc = new RTCPeerConnection(config);

        // connect audio / video
        pc.addEventListener("track", (evt) => {
            if (evt.track.kind == "video") {
            document.getElementById("video").srcObject = evt.streams[0];
            }
        });

        document.getElementById("start").style.display = "none";
        negotiate().then(() => {
            document.getElementById("stop").style.display = "inline-block";
        });
        }

        function stop() {
        document.getElementById("stop").style.display = "none";

        // close peer connection
        setTimeout(() => {
            pc.close();
        }, 500);
        }

            </script>
        </body>
        </html>
        """)  # Only for testing!!!
    
    async def get_bgs(self, r:web.Request) -> web.Response:
        try:
            uid = int((await r.json())["uid"])
        except KeyError:
            return web.HTTPBadRequest(reason="Missing 'uid' query parameters")
        if uid not in self.__processors:
            self.__processors[uid] = self.__Procssor(uid)
        bg_list = self.__processors[uid].get_bg_list()
        return web.json_response([
            {"id": bg.id, "img": self.__encode_img(bg.img)} for bg in bg_list
        ])
    
    async def get_bg(self, r: web.Request) -> web.Response:
        query = await r.json()
        try:
            id = int(query["bgid"])
        except KeyError:
            return web.HTTPBadRequest(reason="Missing 'bgid' query parameters")
        try:
            uid = int(query["uid"])
        except KeyError:
            return web.HTTPBadRequest(reason="Missing 'uid' query parameters")
        return web.Response(content_type="text/plain", text="Not implemented yet")
    
    async def process_offer(self, r: web.Request):
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

        @pc.on("track")
        async def on_track(track):
            print("Track %s received" % track.kind)
            if track.kind == "video":
                print("Setting input track for video processor")
                video.set_track(track)

        # open media source
        pc.addTrack(video)

        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        )

# Example
if __name__ == "__main__":
    class EmptyProcessor(ProcessorABC):
        def __init__(self, uid: int):
            super().__init__(uid)
        
        def process_image(self, img: cv2.typing.MatLike) -> cv2.typing.MatLike:
            return cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

        def set_bg(self, bg_id: int) -> bool:
            if not (bg_id == 1 or bg_id == 2): return False
            self.__bg_id = bg_id
            return True
        
        def get_bg_list(self) -> list[BGData]:
            img1 = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
            img2 = np.random.randint(0, 255, size=(50, 50, 3), dtype=np.uint8)
            return [BGData(1, img1), BGData(2, img2)]
    
    streamer = Streamer(EmptyProcessor)

