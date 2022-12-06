import re
import struct

try:
    import torch
except ImportError:
    print("Tried to import torch, but couldn't. Continuing without it, you might not need it.")

from dt_data_api import DataClient

try:
    from dt_mooc.colab import ColabProgressBar

    _pbar = ColabProgressBar()
    monitor = _pbar.transfer_monitor
except Exception:
    from dt_mooc.utils import plain_progress_monitor as monitor

from dt_mooc.utils import *

from os.path import expanduser


class Storage:

    def __init__(self, token: str, cache_dir: str = None):
        self._client = DataClient(token)
        self._space = self._client.storage("user")
        self._folder = 'courses/mooc/2021/data/nn_models'

        self.cache_directory = cache_dir or "/data"
        if os.path.exists("/code/solution/src"):
            self.cache_directory = "/code/src/"
        if not os.path.exists(self.cache_directory):
            home = expanduser("~")
            self.cache_directory = f"{home}/.dt-nn-models"
        self.cache_directory += "/nn_models"

        if not os.path.exists(self.cache_directory):
            os.makedirs(self.cache_directory)

    @staticmethod
    def export_model(name: str, model, input):
        if re.match('^[0-9a-zA-Z-_.]+$', name) is None:
            raise ValueError("The model name can only container letters, numbers and these "
                             "symbols '.,-,_'")
        # ---
        # export the model
        torch.onnx.export(model,  # model being run
                          input,  # model input (or a tuple for multiple inputs)
                          f"{name}.onnx",
                          # where to save the model (can be a file or file-like object)
                          export_params=True,
                          # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,
                          # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                        'output': {0: 'batch_size'}})

    def upload_yolov5(self, destination_name, pt_model, pt_weights_path):
        # might want to use template pattern if we want to make a bunch of these
        wts_path = pt_weights_path + '.wts'

        # STEP 1: CONVERT TO WTS

        # Get model
        device = select_device('cpu')
        model = pt_model.to(device).float()  # load to FP32
        model.eval()

        # Convert
        with open(wts_path, 'w') as f:
            f.write('{}\n'.format(len(model.state_dict().keys())))
            for k, v in model.state_dict().items():
                vr = v.reshape(-1).cpu().numpy()
                f.write('{} {} '.format(k, len(vr)))
                for vv in vr:
                    f.write(' ')
                    f.write(struct.pack('>f', float(vv)).hex())
                f.write('\n')

        # STEP 2: WRITE HASH
        self.hash(wts_path)
        hash_path = wts_path + ".sha256"

        # STEP 2: UPLOAD BOTH .pt AND .wts
        self._upload(destination_name, [pt_weights_path, wts_path, hash_path])

    def hash(self, filepath, write=True):
        f"""
        Shamelessly stolen from https://stackoverflow.com/a/62214783/11296012
        
        :param filepath: hashes this file
        :return: writes hash to {filepath}.hash and returns the hash
        """
        import hashlib
        import mmap

        hash_filepath = filepath + '.sha256'

        h = hashlib.sha256()
        with open(filepath, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as mm:
                h.update(mm)
        h = h.hexdigest()

        if write:
            with open(hash_filepath, "w") as f:
                f.write(h)

        return h

    # https://github.com/duckietown/lib-dt-data-api/blob/7d53ca7f6dc6b73527c22b3807b21cd33f8e0673/src/dt_data_api/storage.py#L25
    def _upload(self, new_filename, files):
        try:  # promote to iterable if it isn't one
            iter(files)
        except TypeError:
            files = [files]

        for file in files:  # for each file
            dir, old_filename, ext = get_dfe(file)  # split into dir, filename, extension

            # means that we rename but keep the extension!
            destination = os.path.join(self._folder, f"{new_filename}.{ext}")

            print(f'Uploading file `{old_filename + "." + ext}`...')
            handler = self._space.upload(file, destination)
            handler.register_callback(monitor)
            # wait for the upload to finish
            # todo can probably not join after every file, will clean this up if we need a faster version
            handler.join()
            print(
                f'\nFile `{old_filename + "." + ext}` successfully uploaded! It will now be found at `{destination}`.')

    # https://github.com/duckietown/lib-dt-data-api/blob/7d53ca7f6dc6b73527c22b3807b21cd33f8e0673/src/dt_data_api/storage.py#L25
    def _download(self, prefix, destination_directory=None, filter_fun=lambda _: True):
        """
        Downloads all files associated with the given name
        (i.e., all files of that name no matter the extension).

        :return: the downloaded file names
        """
        if destination_directory is None:
            destination_directory = self.cache_directory

        all_files_for_prefix = self._space.list_objects(prefix)
        print("All files for this filename:")
        for file in all_files_for_prefix:
            print(file)
        all_files_for_prefix = list(filter(filter_fun, all_files_for_prefix))
        downloaded_filenames = []

        for file in all_files_for_prefix:  # for each file
            _, old_filename, ext = get_dfe(file)  # split into dir, filename, extension
            full_old_filename = f"{old_filename}.{ext}"
            downloaded_filenames.append(full_old_filename)
            dest = os.path.join(destination_directory, full_old_filename)
            print(f'Downloading file `{full_old_filename}`...')
            handler = self._space.download(file, dest, force=True)
            handler.register_callback(monitor)
            handler.join()
            print(f'\nFile `{full_old_filename}` successfully downloaded! It will now be found at `{dest}`.')

        return downloaded_filenames

    def download_files(self, generic_file_name, destination_directory=None):
        """
        Downloads files to the destination directory. By default, that is our cache, so that future downloads can
        benefit from the cache. On AMD64 though, this is problematic: it means that the users would need to mount the
        cache directory in their dockers. So they can specify a path if they want to.

        :param generic_file_name: shared filename for all files to download
        :param destination_directory:
        :return:
        """
        if not self.is_hash_found_locally(generic_file_name, destination_directory):
            self._download(os.path.join(self._folder, generic_file_name), destination_directory)
            print("As a sanity check, is the hash file now found locally?")

            if self.is_hash_found_locally(generic_file_name, destination_directory):
                print("Sanity check passed.")
            else:
                print("Sanity check failed. Contact us for help.")

        else:
            print(f"Your files were not downloaded because they are already downloaded. "
                  f"You can find them at {os.path.join(self.cache_directory, generic_file_name)}")

    def is_hash_found_locally(self, generic_file_name, cache_directory=None):
        if cache_directory is None:
            cache_directory = self.cache_directory

        if not os.path.exists(os.path.join(cache_directory, generic_file_name + ".sha256")):
            return False

        temp_dir = run("mktemp -d")
        print(f"We will download the hash file to {temp_dir}")
        file_to_download = os.path.join(self._folder, generic_file_name)
        sha_file = self._download(
            file_to_download,
            temp_dir,
            filter_fun=lambda x: x.split("/")[-1] == generic_file_name + ".sha256"

        )
        print("Found sha files:", sha_file)

        if len(sha_file) == 0:
            print("Found no sha file. Something is wrong with your initial upload.")
        assert len(sha_file) == 1, "Found more than one hash in the cloud for your files. Something is wrong"

        with open(os.path.join(temp_dir, sha_file[0]), "r") as f:
            sha = f.read()

        with open(os.path.join(cache_directory, sha_file[0]), "r") as f:
            sha2 = f.read()

        print("Comparing the first hash file (remote),")
        print(f"\t{sha}")
        print(", to the second one (local),")
        print(f"\t{sha2}")

        is_found_locally = sha.strip() == sha2.strip()
        if is_found_locally:
            print(
                f"Equality: found the hash file locally. It is at {os.path.join(cache_directory, sha_file[0])}.")
        else:
            print(f"Non equal: could not find the hash file locally.")
        return is_found_locally

    def upload_model(self, name: str, model, input):
        # export the model
        self.export_model(name, model, input)
        # define source/destination paths
        source = f"{name}.onnx"
        destination = os.path.join(self._folder, f"{name}.onnx")
        # upload the model
        print(f'Uploading model `{name}`...')
        handler = self._space.upload(source, destination)
        handler.register_callback(monitor)
        # wait for the upload to finish
        handler.join()
        print(f'\nModel successfully uploaded!')


if __name__ == "__main__":
    token = sys.argv[1]
    # pt = sys.argv[2]
    store = Storage(token)

    store.download_files("yolov5")

"""
    import sys

    sys.path.insert(0, './yolov5')
    model = torch.load(pt, map_location=select_device("cpu"))['model'].float()  # load to FP32
    model.to(select_device("cpu")).eval()

    store.upload_yolov5("yolov5", model, pt)"""
