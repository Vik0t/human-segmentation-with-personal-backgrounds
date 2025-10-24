from abc import ABC, abstractmethod
from dataclasses import dataclass
import cv2

@dataclass
class BGData:
    id: int
    img: cv2.typing.MatLike


class ProcessorABC(ABC):
    uid: int

    def __init__(self, uid: int):
        self.uid = uid
    
    @abstractmethod
    def process_image(self, img: cv2.typing.MatLike) -> cv2.typing.MatLike:
        ...
    
    @abstractmethod
    def set_bg(self, bg_id: int) -> bool: 
        ...

    @abstractmethod
    def get_bg_list(self) -> list[BGData]:
        ...
    
    @abstractmethod
    def receive_config(self, config: dict) -> bool:
        ...