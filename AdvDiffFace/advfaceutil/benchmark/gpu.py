__all__ = ["GPUDeviceManager"]

from logging import getLogger
from multiprocessing.managers import SyncManager
from typing import Optional

from torch import device
from torch.cuda import device_count as cuda_device_count


LOGGER = getLogger("gpu_manager")


class GPUDeviceManager:
    def __init__(self, manager: SyncManager, *, processes_per_gpu: int = 1):
        self._device_count = cuda_device_count()
        self._processes_per_gpu = processes_per_gpu

        # Use a queue to store the available devices
        self._device_queue = manager.Queue(
            maxsize=self._device_count * processes_per_gpu
        )

        # Add all the GPUs to the queue
        for gpu in range(self._device_count):
            for _ in range(self._processes_per_gpu):
                self._device_queue.put_nowait(gpu)

        self._acquired_device_index: Optional[int] = None
        self._acquired_device: Optional[device] = None

    @property
    def device_count(self) -> int:
        return self._device_count

    @property
    def processes_per_gpu(self) -> int:
        return self._processes_per_gpu

    def acquire(self) -> device:
        if self._acquired_device is not None:
            return self._acquired_device

        # Get a device from the queue (waiting if one is not available)
        self._acquired_device_index = self._device_queue.get()
        self._acquired_device = device("cuda:%d" % self._acquired_device_index)

        LOGGER.info("Acquired device %d", self._acquired_device_index)

        return self._acquired_device

    def release(self):
        # Put the acquired device back on the queue
        self._device_queue.put_nowait(self._acquired_device_index)

        LOGGER.info("Released device %d", self._acquired_device_index)

        # Reset our acquired device
        self._acquired_device_index = None
        self._acquired_device = None
