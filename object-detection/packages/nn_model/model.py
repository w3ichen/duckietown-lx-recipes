import ctypes
import os
import tempfile
from typing import Tuple

from solution.integration_activity import MODEL_NAME, DT_TOKEN

from dt_device_utils import DeviceHardwareBrand, get_device_hardware_brand
from dt_mooc.cloud import Storage

from .constants import ASSETS_DIR, IMAGE_SIZE


USE_FP16 = True


def run(input, exception_on_failure=False):
    print(input)
    try:
        import subprocess

        program_output = subprocess.check_output(
            f"{input}", shell=True, universal_newlines=True, stderr=subprocess.STDOUT
        )
    except Exception as e:
        if exception_on_failure:
            print(e.output)
            raise e
        program_output = e.output
    print(program_output)
    return program_output.strip()


class Wrapper:
    def __init__(self, aido_eval=False):
        model_name = MODEL_NAME()

        models_path = os.path.join(ASSETS_DIR, "nn_models")
        weight_file_path = os.path.join(models_path, model_name)

        if aido_eval:
            assert os.path.exists(weight_file_path)
            self.model = AMD64Model(weight_file_path)
            return

        dt_token = DT_TOKEN()

        if get_device_hardware_brand() == DeviceHardwareBrand.JETSON_NANO:
            # when running on the robot, we store models in the persistent `data` directory
            models_path = "/data/nn_models"
            weight_file_path = os.path.join(models_path, model_name)

        # make models destination dir if it does not exist
        if not os.path.exists(models_path):
            os.makedirs(models_path)

        # open a pointer to the DCSS storage unit
        storage = Storage(dt_token, cache_dir=models_path)

        # do not download if already up-to-date
        model_exists = storage.is_hash_found_locally(model_name, models_path)
        if not model_exists:
            storage.download_files(model_name, models_path)

        # TODO: during MOOC2021, we needed to convert a .pt model to .wts and then tensorRT on the JN 2GB
        # if False:
        #     if get_device_hardware_brand() == DeviceHardwareBrand.JETSON_NANO and not file_already_existed:
        #         print("\n\n\n\nCONVERTING TO ONNX. THIS WILL TAKE A LONG TIME...\n\n\n")
        #         # https://github.com/duckietown/tensorrtx/tree/dt-yolov5/yolov5
        #         run("git clone https://github.com/duckietown/tensorrtx.git -b dt-obj-det")
        #         run(f"cp {weight_file_path}.wts ./tensorrtx/yolov5.wts")
        #         run(
        #             f"cd tensorrtx && ls && chmod 777 ./do_convert.sh && ./do_convert.sh",
        #             exception_on_failure=True,
        #         )
        #         run(f"mv tensorrtx/build/yolov5.engine {weight_file_path}.engine")
        #         run(f"mv tensorrtx/build/libmyplugins.so {weight_file_path}.so")
        #         run("rm -rf tensorrtx")
        #         print(
        #             "\n\n\n\n...DONE CONVERTING! NEXT TIME YOU RUN USING THE SAME MODEL, WE WON'T NEED TO DO THIS!\n\n\n"
        #         )
        #
        #     if get_device_hardware_brand() == DeviceHardwareBrand.JETSON_NANO:
        #         self.model = TRTModel(weight_file_path)
        #
        #     else:
        #         self.model = AMD64Model(weight_file_path)

        # TODO: during MOOC2022, we want to try and use the .pt model directly on the JN 4GB
        self.model = AMD64Model(weight_file_path)

    def predict(self, image):
        return self.model.infer(image)


class Model:
    def __init__(self):
        pass

    def infer(self, image):
        raise NotImplementedError()


class AMD64Model:
    def __init__(self, weight_file_path):
        super().__init__()

        import torch

        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.hub.set_dir(tmpdirname)
            model = torch.hub.load("ultralytics/yolov5", "custom", path=f"{weight_file_path}.pt")

        if USE_FP16:
            model = model.half()

        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()

        del model

    def infer(self, image) -> Tuple[list, list, list]:
        det = self.model(image, size=IMAGE_SIZE)

        xyxy = det.xyxy[0]  # grabs det of first image (aka the only image we sent to the net)

        if xyxy.shape[0] > 0:
            conf = xyxy[:, -2]
            clas = xyxy[:, -1]
            xyxy = xyxy[:, :-2]

            return xyxy.tolist(), clas.tolist(), conf.tolist()
        return [], [], []


# # TODO: we might be able to get rid of this completely if the model works on the JN 4GB without tensorRT
# class TRTModel(Model):
#     def __init__(self, weight_file_path):
#         super().__init__()
#         ctypes.CDLL(weight_file_path + ".so")
#         from object_detection.tensorrt_model import YoLov5TRT
#
#         self.model = YoLov5TRT(weight_file_path + ".engine")
#
#     def infer(self, image):
#         # todo ensure this is in boxes, classes, scores format
#         results = self.model.infer_for_robot([image])
#         boxes = results[0][0]
#         confs = results[0][1]
#         classes = results[0][2]
#
#         if classes.shape[0] > 0:
#             return boxes, classes, confs
#         return [], [], []
