import os, subprocess, json, sys
import threading, signal
from pathlib import Path
import fiftyone as fo


class SessionThread(threading.Thread):
    def __init__(self):
        super(SessionThread, self).__init__()

    def start(self, cmd):
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True
        )

    def stop(self):
        # print(self.process)
        # self.process.send_signal(signal.SIGINT)
        # self.process.send_signal(signal.SIGINT)
        # self.process.send_signal(signal.SIGKILL)
        self.process.terminate()
        # TODO: none of the above methods seem to work to unmount

        return_code = self.process.wait()
        print(f"Process terminated with exit code {return_code}")


class LuxonisSession:
    def __init__(self):
        self.mount_path = f"{str(Path.home())}/.luxonis_mount"
        os.makedirs(self.mount_path, exist_ok=True)

        credentials_cache_file = (
            f"{str(Path.home())}/.cache/luxonis_ml/credentials.json"
        )
        if os.path.exists(credentials_cache_file):
            with open(credentials_cache_file) as file:
                self.creds = json.load(file)
        else:
            self.creds = {}

    def _get_credentials(self, key):
        if key in self.creds.keys():
            return self.creds[key]
        else:
            return os.environ[key]

    def _get_cmd(self):
        return f"s3fs luxonis-test-bucket {self.mount_path} -o passwd_file={str(Path.home())}/.passwd-s3fs -o curldbg -o url={self._get_credentials('AWS_S3_ENDPOINT_URL')} -o use_path_request_style"

    def _app_worker(self, dataset):
        app_session = fo.launch_app(dataset.fo_dataset)

        while True:
            pass

    def start(self):
        cmd = self._get_cmd()
        self.thread = SessionThread()
        self.thread.start(cmd)

    def stop(self):
        self.thread.stop()

    def launch_app(self, dataset):
        self.app_thread = threading.Thread(target=self._app_worker, args=[dataset])
        self.app_thread.start()
