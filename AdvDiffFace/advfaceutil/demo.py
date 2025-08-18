from argparse import ArgumentParser
from ctypes import windll
from functools import partial
from logging import getLogger
from pathlib import Path
from time import strftime
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import *
from typing import List
from typing import Optional
from typing import Tuple

import cv2
import numpy as np
from PIL import Image as PImage
from PIL import ImageTk
from pyfacear import OBJMeshIO
from torch import device
from torch.cuda import is_available as cuda_is_available

from advfaceutil.benchmark import Accessory
from advfaceutil.benchmark import AngleVariance
from advfaceutil.benchmark import AngleVarianceProperties
from advfaceutil.benchmark import Benchmark
from advfaceutil.benchmark import BenchmarkArguments
from advfaceutil.benchmark import BenchmarkData
from advfaceutil.benchmark import construct_all
from advfaceutil.benchmark import DataProperty
from advfaceutil.benchmark import Statistic
from advfaceutil.benchmark import SuccessRate
from advfaceutil.benchmark import TopK
from advfaceutil.datasets import FaceDatasets
from advfaceutil.recognition import RecognitionArchitecture
from advfaceutil.recognition import RecognitionArchitectures
from advfaceutil.recognition.processing import AugmentationOptions
from advfaceutil.recognition.processing import DlibAugmentationOptions
from advfaceutil.recognition.processing import FaceProcessors
from advfaceutil.recognition.processing import MediaPipeAugmentationOptions
from advfaceutil.ui import Camera
from advfaceutil.ui import EnumOptionMenu
from advfaceutil.ui import EnumVar
from advfaceutil.ui import trace
from advfaceutil.utils import load_overlay
from advfaceutil.utils import SetLogLevel
from advfaceutil.utils import to_pretty_json

LOGGER = getLogger("demo")

CONFIDENCE = float("-inf")

# Fix the UI on high DPI displays
windll.shcore.SetProcessDpiAwareness(1)


DEVICE = device("cuda" if cuda_is_available() else "cpu")


DEFAULT_DEMO_BENCHMARK_STATISTICS = [
    AngleVariance.Factory(),
    SuccessRate.Factory(),
    TopK.Factory(3),
]


DEFAULT_DEMO_BENCHMARK_BIN_PROPERTY_COMBINATIONS = [
    (AngleVarianceProperties.PITCH.with_sized_bin(10),),
    (AngleVarianceProperties.YAW.with_sized_bin(10),),
    (
        AngleVarianceProperties.PITCH.with_sized_bin(10),
        AngleVarianceProperties.YAW.with_sized_bin(10),
    ),
]


def _path_to_string(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    return path.as_posix()


class DemoBenchmark(Benchmark):
    def __init__(
        self,
        benchmark_arguments: BenchmarkArguments,
        accessory: Accessory,
        statistics: List[Statistic],
        bin_property_combinations: List[Tuple[DataProperty, ...]],
    ):
        super().__init__(
            "demo",
            benchmark_arguments,
            accessory,
            [],
            statistics,
            [],
            bin_property_combinations,
        )

        self._processed_data = []

    def process(
        self,
        image: np.ndarray,
        augmented: Optional[np.ndarray],
        aligned: np.ndarray,
        augmented_aligned: Optional[np.ndarray],
        architecture: RecognitionArchitecture,
    ) -> None:
        data = BenchmarkData(
            image,
            self._accessory.base_class,
            self._accessory.base_class_index,
            Path("camera"),
            augmented if augmented is not None else image,
            aligned,
            augmented_aligned if augmented_aligned is not None else aligned,
        )

        # Classify the data
        self._classify_data(data, architecture)

        # Record the statistics
        self._record_statistics(data)

        self._processed_data.append(data.compress())

    def save(self) -> Path:
        self._collate_statistics(self._processed_data)

        self._save_config()

        if self._benchmark_arguments.save_raw_statistics:
            (self._benchmark_arguments.output_directory / "raw.json").write_text(
                to_pretty_json(
                    list(map(lambda d: d.to_json_dict(), self._processed_data))
                )
            )

        self._processed_data.clear()

        return self._output_directory


class BenchmarkSettings(Frame):
    def __init__(self, master: Optional[Misc], demo: "Demo"):
        super().__init__(master)
        self.demo = demo

        self.demo_benchmark: Optional[DemoBenchmark] = None

        size = self.demo.recognition.dataset.get().get_size(
            self.demo.recognition.dataset_size.get()
        )

        self.base_class = StringVar(self, name="base_class", value=size.class_names[0])
        self.target_class = StringVar(self, name="target_class", value="None")

        self.save_augmented_images = BooleanVar(self, False, "save_augmented_images")
        self.save_aligned_images = BooleanVar(self, False, "save_aligned_images")

        self.base_class_label = Label(self, text="Base Class")
        self.base_class_label.grid_configure(row=0, column=0, sticky=W)
        self.base_class_options = OptionMenu(
            self,
            self.base_class,
            self.base_class.get(),
            *size.class_names,
        )
        self.base_class_options.grid_configure(row=0, column=1, sticky=W + E)

        self.target_class_label = Label(self, text="Target Class")
        self.target_class_label.grid_configure(row=1, column=0, sticky=W)
        self.target_class_options = OptionMenu(
            self,
            self.target_class,
            self.target_class.get(),
            "None",
            *self.demo.recognition.dataset.get()
            .get_size(self.demo.recognition.dataset_size.get())
            .class_names,
        )
        self.target_class_options.grid_configure(row=1, column=1, sticky=W + E)

        self.save_augmented_images_button = Checkbutton(
            self, text="Save Augmented Images", variable=self.save_augmented_images
        )
        self.save_augmented_images_button.grid_configure(
            row=2, column=0, columnspan=2, sticky=W + E
        )

        self.save_aligned_images_button = Checkbutton(
            self, text="Save Aligned Images", variable=self.save_aligned_images
        )
        self.save_aligned_images_button.grid_configure(
            row=3, column=0, columnspan=2, sticky=W + E
        )

        self.toggle_benchmark_button = Button(
            self, text="Start Benchmark", command=self._toggle_benchmark
        )
        self.toggle_benchmark_button.grid_configure(
            row=4, column=0, columnspan=2, sticky=W + E
        )
        self.toggle_benchmark_button.config(state=DISABLED)

    @property
    def is_benchmark_running(self) -> bool:
        return self.demo_benchmark is not None

    def _toggle_benchmark(self):
        if not self.is_benchmark_running:
            size = self.demo.recognition.dataset.get().get_size(
                self.demo.recognition.dataset_size.get()
            )

            # Set the output directory to be a sub-folder of the output directory for the
            # current date and time
            output_directory = Path("output") / strftime("%Y-%m-%d_%H-%M-%S")
            # Create the directory if it doesn't exist
            output_directory.mkdir(parents=True, exist_ok=True)

            benchmark_arguments = BenchmarkArguments(
                # The dataset directories are not used
                Path("."),
                Path("."),
                output_directory,
                self.demo.recognition.recognition_architecture.get(),
                self.demo.recognition.dataset.get(),
                size,
                self.demo.face_processing.face_processor_type.get(),
                self.demo.face_processing.additive_mask.get(),
                Path(self.demo.face_processing.mask_model_path.get()),
                save_augmented_images=self.save_augmented_images.get(),
                save_aligned_images=self.save_aligned_images.get(),
            )

            accessory = Accessory(
                Path(self.demo.face_processing.mask_path.get()),
                Path("invalid"),
                self.base_class.get(),
                self.target_class.get() if self.target_class.get() != "None" else None,
                size.class_names.index(self.base_class.get()),
                size.class_names.index(self.target_class.get())
                if self.target_class.get() != "None"
                else None,
            )

            self.demo_benchmark = DemoBenchmark(
                benchmark_arguments,
                accessory,
                construct_all(
                    DEFAULT_DEMO_BENCHMARK_STATISTICS, benchmark_arguments, accessory
                ),
                DEFAULT_DEMO_BENCHMARK_BIN_PROPERTY_COMBINATIONS,
            )

            self.toggle_benchmark_button.configure(text="Stop Benchmark")
        else:
            self.demo_benchmark.save()
            self.demo.add_status(
                f"Benchmark saved to {self.demo_benchmark.benchmark_arguments.output_directory.as_posix()}"
            )
            self.demo_benchmark = None
            self.toggle_benchmark_button.configure(text="Start Benchmark")


class RecognitionSettings(LabelFrame):
    def __init__(self, master: Optional[Misc], demo: "Demo", *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        self.demo = demo

        self.dataset = EnumVar(self, FaceDatasets, FaceDatasets.PUBFIG, "dataset")
        self.dataset_size = StringVar(self, "SMALL", "dataset_size")
        self.recognition_architecture = EnumVar(
            self,
            RecognitionArchitectures,
            RecognitionArchitectures.VGG,
            "recognition_architecture",
        )
        self.recognition_enabled = BooleanVar(self, False, "recognition_enabled")

        self.recognition = self.recognition_architecture.get().construct(
            self.dataset.get(),
            self.dataset.get().get_size(self.dataset_size.get()),
            device=DEVICE,
        )

        self.recognition_button = Checkbutton(
            self,
            text="Enable recognition",
            variable=self.recognition_enabled,
            state=DISABLED,
        )
        self.recognition_button.grid_configure(row=1, column=0, columnspan=2, sticky=W)
        Label(self, text="Dataset").grid_configure(row=2, column=0, sticky=W)
        self.dataset_options = EnumOptionMenu(
            self,
            self.dataset,
            FaceDatasets,
        )
        self.dataset_options.configure(state=DISABLED)
        self.dataset_options.grid_configure(row=2, column=1, sticky=W)
        Label(self, text="Dataset Size").grid_configure(row=3, column=0, sticky=W)
        self.dataset_size_options = OptionMenu(
            self,
            self.dataset_size,
            self.dataset_size.get(),
            "SMALL",
            "LARGE",
        )
        self.dataset_size_options.configure(state=DISABLED)
        self.dataset_size_options.grid_configure(row=3, column=1, sticky=W)

        Label(self, text="Recognition Architecture").grid_configure(
            row=4, column=0, sticky=W
        )
        self.recognition_architecture_options = EnumOptionMenu(
            self,
            self.recognition_architecture,
            RecognitionArchitectures,
        )
        self.recognition_architecture_options.configure(state=DISABLED)
        self.recognition_architecture_options.grid_configure(row=4, column=1, sticky=W)

    @trace("recognition_enabled", "write")
    def _update_recognition_enabled(self):
        enabled = self.recognition_enabled.get()
        if enabled:
            self.dataset_options.config(state=NORMAL)
            self.dataset_size_options.config(state=NORMAL)
            self.recognition_architecture_options.config(state=NORMAL)

            if self.demo.face_processing.alignment.get():
                self.demo.benchmark_settings.toggle_benchmark_button.config(
                    state=NORMAL
                )
        else:
            self.dataset_options.config(state=DISABLED)
            self.dataset_size_options.config(state=DISABLED)
            self.recognition_architecture_options.config(state=DISABLED)

            self.demo.benchmark_settings.toggle_benchmark_button.config(state=DISABLED)

    @trace("dataset", "write")
    def __change_dataset(self):
        # todo: check if it has changed
        self.recognition = self.recognition_architecture.get().construct(
            self.dataset.get(),
            self.dataset.get().get_size(self.dataset_size.get()),
            device=DEVICE,
        )

    @trace("dataset_size", "write")
    def __change_dataset_size(self):
        # todo: check if it has changed
        self.recognition = self.recognition_architecture.get().construct(
            self.dataset.get(),
            self.dataset.get().get_size(self.dataset_size.get()),
            device=DEVICE,
        )

    @trace("recognition_architecture", "write")
    def __change_architecture(self):
        # todo: check if it has changed
        self.recognition = self.recognition_architecture.get().construct(
            self.dataset.get(),
            self.dataset.get().get_size(self.dataset_size.get()),
            device=DEVICE,
        )


class FaceProcessingSettings(LabelFrame):
    def __init__(
        self,
        master: Optional[Misc],
        demo: "Demo",
        *args,
        **kwargs,
    ):
        super().__init__(master, *args, **kwargs)
        self.demo = demo

        self.face_processor_type = EnumVar(
            self, FaceProcessors, FaceProcessors.MEDIAPIPE, "face_processor"
        )
        self.alignment = BooleanVar(self, False, "alignment")
        self.alignment_bb = BooleanVar(self, True, "alignment_bb")
        self.visualise_alignment = BooleanVar(self, False, "visualise_alignment")
        self.enable_mask = BooleanVar(self, False, "enable_mask")
        self.mask_path = StringVar(self, name="mask_path")
        self.mask_model_path = StringVar(self, name="mask_model_path")
        self.additive_mask = BooleanVar(self, False, "additive_mask")

        self.face_processor = self.face_processor_type.get().construct()
        self.mask_image = None
        self.mesh = None
        self.augmentation_options: Optional[AugmentationOptions] = None

        Label(self, text="Face Processor").grid_configure(row=0, column=0, sticky=W)
        self.face_processor_options = EnumOptionMenu(
            self,
            self.face_processor_type,
            FaceProcessors,
        )
        self.face_processor_options.grid_configure(row=0, column=1, sticky=W)
        self.alignment_button = Checkbutton(
            self, text="Enable alignment", variable=self.alignment
        )
        self.alignment_button.grid_configure(row=1, column=0, columnspan=2, sticky=W)
        self.alignment_bb_button = Checkbutton(
            self, text="Show bounding box", variable=self.alignment_bb, state=DISABLED
        )
        self.alignment_bb_button.grid_configure(row=2, column=0, columnspan=2, sticky=W)
        self.visualise_alignment_button = Checkbutton(
            self,
            text="Visualise alignment",
            variable=self.visualise_alignment,
            state=DISABLED,
        )
        self.visualise_alignment_button.grid_configure(
            row=3, column=0, columnspan=2, sticky=W
        )
        self.enable_mask_button = Checkbutton(
            self, text="Enable mask", variable=self.enable_mask, state=DISABLED
        )
        self.enable_mask_button.grid_configure(row=4, column=0, columnspan=2, sticky=W)

        self.open_mask_button = Button(self, text="Open Mask", command=self._open_mask)
        self.open_mask_button.grid_configure(
            row=8, column=0, columnspan=2, sticky=W + E
        )
        self.mask_image_label = Label(self)
        self.mask_image_label.grid_configure(
            row=9, column=0, columnspan=2, sticky=W + E
        )
        self.additive_mask_button = Checkbutton(
            self, text="Additive mask", variable=self.additive_mask, state=DISABLED
        )
        self.additive_mask_button.grid_configure(
            row=10, column=0, columnspan=2, sticky=W
        )
        self.show_alignment_button = Button(
            self,
            text="Show Aligned Image",
            command=self.demo.show_alignment_window,
            state=DISABLED,
        )
        self.show_alignment_button.grid_configure(
            row=11, column=0, columnspan=2, sticky=W + E
        )
        self.open_mask_button = Button(
            self, text="Open Model", command=self._open_model
        )
        self.open_mask_button.grid_configure(
            row=12, column=0, columnspan=2, sticky=W + E
        )

    def _open_mask(self):
        file_path = filedialog.askopenfilename(
            title="Open Mask",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("Numpy files", "*.npy"),
            ],
        )
        if file_path:
            self.mask_path.set(file_path)

    def _open_model(self):
        file_path = filedialog.askopenfilename(
            title="Open Model",
            filetypes=[
                ("OBJ files", "*.obj"),
            ],
        )
        if file_path:
            self.mask_model_path.set(file_path)

    @trace("mask_path", "write")
    def _update_mask_path(self):
        mask_path = self.mask_path.get()
        self.mask_image = load_overlay(mask_path)

        # Resize the mask image to fit the size of this widget
        width = self.winfo_width() - 10
        height = int(self.mask_image.shape[0] * width / self.mask_image.shape[1])
        new_size = (width, height)
        image = ImageTk.PhotoImage(PImage.fromarray(self.mask_image).resize(new_size))

        # Ensure the mask image is a float32
        self.mask_image = self.mask_image.astype(np.float32)

        if self.augmentation_options is None:
            if self.face_processor_type.get() == FaceProcessors.MEDIAPIPE:
                self.augmentation_options = MediaPipeAugmentationOptions(
                    texture=self.mask_image.astype(np.uint8), mesh=self.mesh
                )
            else:
                self.augmentation_options = DlibAugmentationOptions(
                    texture=self.mask_image.astype(np.uint8),
                    additive_overlay=self.additive_mask.get(),
                )
        else:
            self.augmentation_options.texture = self.mask_image.astype(np.uint8)

        self.mask_image_label.image = image
        self.mask_image_label.configure(image=image)
        self.additive_mask_button.configure(state=NORMAL)
        if self.alignment.get():
            self.enable_mask_button.configure(state=NORMAL)

    @trace("mask_model_path", "write")
    def _update_mask_model_path(self):
        if self.face_processor_type.get() == FaceProcessors.MEDIAPIPE:
            self.mesh = OBJMeshIO.load(self.mask_model_path.get())
            self.mesh.flip_texture_coords()
            if self.augmentation_options is None:
                self.augmentation_options = MediaPipeAugmentationOptions(
                    texture=self.mask_image.astype(np.uint8), mesh=self.mesh
                )
            else:
                self.augmentation_options.mesh = self.mesh

    @trace("additive_mask", "write")
    def _update_additive_mask(self):
        if self.face_processor_type.get() == FaceProcessors.DLIB:
            if self.augmentation_options is None:
                self.augmentation_options = DlibAugmentationOptions(
                    texture=self.mask_image.astype(np.uint8),
                    additive_overlay=self.additive_mask.get(),
                )
            else:
                self.augmentation_options.additive_overlay = self.additive_mask.get()

    @trace("face_processor_type", "write")
    def _update_face_processor(self):
        self.augmentation_options = None
        self.face_processor = self.face_processor_type.get().construct()

    @trace("alignment", "write")
    def _update_alignment(self):
        enabled = self.alignment.get()
        if enabled:
            self.alignment_bb_button.config(state=NORMAL)
            self.visualise_alignment_button.config(state=NORMAL)
            self.demo.recognition.recognition_button.config(state=NORMAL)
            self.show_alignment_button.config(state=NORMAL)
            if self.mask_image is not None:
                self.enable_mask_button.config(state=NORMAL)

            if self.demo.recognition.recognition_enabled.get():
                self.demo.benchmark_settings.toggle_benchmark_button.config(
                    state=NORMAL
                )
        else:
            self.alignment_bb_button.config(state=DISABLED)
            self.show_alignment_button.config(state=DISABLED)
            self.visualise_alignment_button.config(state=DISABLED)
            self.demo.recognition.recognition_button.config(state=DISABLED)
            self.enable_mask_button.config(state=DISABLED)

            self.demo.benchmark_settings.toggle_benchmark_button.config(state=DISABLED)


class AlignmentPopup(Toplevel):
    def __init__(self, master: "Demo"):
        super().__init__(master)
        self.wm_transient(master)
        self.wm_resizable(False, False)
        self.title("Aligned Images")
        self.canvas = Canvas(self, width=224, height=224, background="black")
        self.canvas.pack()

        self.__image_output = None

    def update_image(self, image: np.ndarray) -> bool:
        if not self.winfo_exists():
            return False
        image = ImageTk.PhotoImage(image=PImage.fromarray(image))

        if self.__image_output is not None:
            try:
                self.canvas.itemconfigure(self.__image_output, image=image)
            except TclError:
                self.__image_output = self.canvas.create_image(
                    0, 0, image=image, anchor=NW
                )
        else:
            self.__image_output = self.canvas.create_image(0, 0, image=image, anchor=NW)
        self.canvas.image = image
        return True


class Demo(Tk):
    def __init__(self):
        super().__init__()

        self.title("AdvDiffFace Demo")
        self.wm_geometry("1000x600")

        self.main_content = Frame(self)
        self.main_content.pack_configure(
            padx=20, pady=(20, 0), fill=BOTH, expand=YES, side=TOP
        )

        self.config_panel = Frame(self.main_content)
        self.config_panel.pack_configure(padx=(10, 0), fill=Y, side=RIGHT)
        self.config_panel.grid_rowconfigure(2, weight=1)
        self.config_panel.grid_columnconfigure(0, weight=1)

        self.config_tabs = Notebook(self.config_panel)
        self.config_tabs.pack_configure(fill=BOTH, expand=YES, side=TOP)

        self.face_settings = Frame(self.config_tabs)

        self.recognition = RecognitionSettings(
            self.face_settings,
            self,
            text="Recognition Settings",
            padding=5,
        )
        self.recognition.grid(row=0, column=0, sticky=W + E, pady=(0, 5))
        self.recognition.grid_columnconfigure(0, weight=1)
        self.face_processing = FaceProcessingSettings(
            self.face_settings,
            self,
            text="Face Processing Settings",
            padding=5,
        )
        self.face_processing.grid(row=1, column=0, sticky=W + E, pady=(0, 5))
        self.face_processing.grid_columnconfigure(0, weight=1)

        self.config_tabs.add(self.face_settings, text="Processing")

        self.benchmark_settings = BenchmarkSettings(self.config_tabs, self)
        self.config_tabs.add(self.benchmark_settings, text="Benchmark")

        self.capture_photo_button = Button(
            self.config_panel, text="Capture Photo", command=self._capture_photo
        )
        self.capture_photo_button.pack_configure(
            fill=X, side=BOTTOM, expand=NO, pady=(0, 5)
        )

        self.camera = Camera(self.main_content, lambda image: self._recognise(image))
        self.camera.pack_configure(
            expand=YES, fill=BOTH, side=LEFT, before=self.config_panel
        )

        self.status_bar = Label(self)
        self.status_bar.pack_configure(
            padx=20,
            pady=(0, 5),
            fill=X,
            side=BOTTOM,
            expand=NO,
            before=self.main_content,
        )

        self.menubar = Menu(self)
        self.config(menu=self.menubar)
        self.__create_menubar()

        self.alignment_popup = None

        self.bind("<Visibility>", self._on_load)
        self.wm_protocol("WM_DELETE_WINDOW", self._on_close)

    def show_alignment_window(self):
        if self.alignment_popup is not None and self.alignment_popup.winfo_exists():
            return
        self.face_processing.show_alignment_button.config(state=DISABLED)
        self.alignment_popup = AlignmentPopup(self)

    def _recognise(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        if not self.face_processing.alignment.get():
            return image

        faces = self.face_processing.face_processor.detect_faces(image)

        if faces is None:
            return image

        base_image = image

        augmentation_enabled = (
            self.face_processing.enable_mask.get()
            and self.face_processing.augmentation_options is not None
        )

        if augmentation_enabled:
            base_image = image.copy()
            image = self.face_processing.face_processor.augment(
                image, self.face_processing.augmentation_options, faces
            )

        recognition_architecture = self.recognition.recognition_architecture.get()

        aligned_faces = self.face_processing.face_processor.align(
            image, recognition_architecture.crop_size, faces
        )

        if aligned_faces is None:
            return image

        if (
            self.benchmark_settings.is_benchmark_running
            and len(faces) == 1
            and len(aligned_faces) == 1
            and self.recognition.recognition_enabled.get()
        ):
            if augmentation_enabled:
                self.benchmark_settings.demo_benchmark.process(
                    base_image,
                    image,
                    self.face_processing.face_processor.align(
                        base_image, self.recognition.recognition.crop_size, faces
                    )[0],
                    aligned_faces[0],
                    self.recognition.recognition,
                )
            else:
                self.benchmark_settings.demo_benchmark.process(
                    base_image,
                    None,
                    aligned_faces[0],
                    None,
                    self.recognition.recognition,
                )

        # Align with landmark detection and add bounding box visualisation
        for face, aligned in zip(faces, aligned_faces):
            if self.alignment_popup is not None:
                if not self.alignment_popup.update_image(aligned):
                    self.face_processing.show_alignment_button.configure(state=NORMAL)

            if self.recognition.recognition_enabled.get():
                logits = self.recognition.recognition.logits(aligned, DEVICE)
                if logits.max() >= CONFIDENCE:
                    clazz = logits.argmax().item()
                    class_name = (
                        self.recognition.dataset.get()
                        .get_size(self.recognition.dataset_size.get())
                        .class_names[clazz]
                    )
                    image = cv2.putText(
                        image,
                        class_name + " %.2f" % logits.max(),
                        (face.bounding_box.left, face.bounding_box.top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

        if self.face_processing.alignment_bb.get():
            self.face_processing.face_processor.show_bounding_boxes(
                image, faces, thickness=2
            )
        if self.face_processing.visualise_alignment.get():
            self.face_processing.face_processor.show_landmarks(image, faces, radius=1)

        return image

    def add_status(self, text: str):
        self.status_bar.configure(text=text)
        self.status_bar.after(5000, lambda: self.status_bar.configure(text=""))

    def _capture_photo(self):
        if path := self.camera.capture_photo():
            self.add_status(f"Saved photo at {path.as_posix()}")

    def _on_load(self, _):
        self.unbind("<Visibility>")
        self.camera.start_camera()

    def _on_close(self):
        self.camera.stop_camera()
        self.destroy()

    def __configure_camera(self):
        window = Toplevel()
        window.wm_transient(self)
        window.title("Camera Settings")

        frame = LabelFrame(window, text="Camera")

        resolution_menu = OptionMenu(
            frame,
            self.camera.selected_resolution,
            self.camera.selected_resolution.get(),
            *self.camera.available_resolutions,
        )
        resolution_menu.grid(row=0, column=1, sticky=W + E)
        Label(frame, text="Resolution").grid(row=0, column=0, sticky=W + E)

        # noinspection PyTypeChecker
        framerate_menu = OptionMenu(
            frame,
            self.camera.selected_framerate,
            self.camera.selected_framerate.get(),
            *self.camera.available_framerates,
        )
        framerate_menu.grid(row=1, column=1, sticky=W + E)
        Label(frame, text="Framerate").grid(row=1, column=0, sticky=W + E)

        frame.pack_configure(fill=BOTH, expand=YES, padx=10, pady=10)

        window.update()
        window.wm_minsize(window.winfo_width(), window.winfo_height())
        window.geometry(
            f"+{self.winfo_x() + (self.winfo_width() - window.winfo_width()) // 2}+{self.winfo_y() + (self.winfo_height() - window.winfo_height()) // 2}"
        )

    @trace("camera.is_active", "write")
    def _update_camera_is_active(self):
        if self.camera.is_active.get():
            self.file_menu.entryconfigure("Start camera", state=DISABLED)
            self.file_menu.entryconfigure("Stop camera", state=NORMAL)
        else:
            self.file_menu.entryconfigure("Start camera", state=NORMAL)
            self.file_menu.entryconfigure("Stop camera", state=DISABLED)

    def __create_menubar(self):
        self.file_menu = Menu(self.menubar, tearoff=False)

        is_active = self.camera.is_active.get()
        self.file_menu.add_command(
            label="Start camera",
            command=self.camera.start_camera,
            state=DISABLED if is_active else NORMAL,
        )
        self.file_menu.add_command(
            label="Stop camera",
            command=self.camera.stop_camera,
            state=NORMAL if is_active else DISABLED,
        )
        self.file_menu.add_command(
            label="Camera Settings", command=self.__configure_camera
        )
        self.file_menu.add_command(label="Quit", command=self.destroy)

        self.menubar.add_cascade(label="File", menu=self.file_menu)

        devices_menu = Menu(self.menubar, tearoff=False)
        for device_index, name in self.camera.available_cameras.items():
            switch = partial(self.camera.switch_camera, device_index)
            devices_menu.add_radiobutton(
                label=f"{device_index}: {name}",
                command=switch,
                value=device_index,
                variable=self.camera.selected_camera,
            )

        self.menubar.add_cascade(label="Devices", menu=devices_menu)


if __name__ == "__main__":
    # Use an argument parser to allow setting the log level
    parser = ArgumentParser()
    SetLogLevel.add_args(parser)
    args = parser.parse_args()
    SetLogLevel.parse_args(args)

    demo = Demo()
    demo.mainloop()