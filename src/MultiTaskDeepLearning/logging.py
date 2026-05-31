

import os
import shutil
from datetime import datetime
import sys


class Logging:
    """
    A class for handling logging directories and copying configuration files.

    Attributes:
        log_dir (str): The base directory for logging.
        dataset_model (str): A formatted string combining dataset and model names.
        info (str): Additional information for log directory naming.
        log_directory (str): The computed directory where logs and files will be stored.
    """

    def __init__(self,
                 log_dir: str,
                 redirect_output: bool = False) -> None:
        """
        Initializes the Logging class by setting up the log directory and copying files.

        Args:
            log_dir (str): The base directory for storing logs.
            dataset (str): The dataset name.
            model (str): The model name.
            info (str): Additional information for directory naming.
            files_to_copy (List[str]): A list of file paths to copy into the log directory.
        """
        self.log_dir = log_dir
        self.dataset_model = f"{dataset}-{model}"
        self.info = info

        self.log_directory = self._log_directory_name()

        os.makedirs(self.log_directory, exist_ok=True)

        if redirect_output:
            print(f"Log directory: {self.log_directory}")
            log_file = os.path.join(self.log_directory, "output.log")
            sys.stdout = open(log_file, 'a')
            sys.stderr = sys.stdout  # Also redirect stderr to the same log

    def _log_directory_name(self) -> str:
        """
        Generates a unique log directory name, appending a timestamp if needed.

        Returns:
            str: The full path of the log directory.
        """
        base_path = os.path.join(self.log_dir, self.dataset_model, self.info)
        if os.path.exists(base_path):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M:%S")
            base_path = f"{base_path}_{timestamp}"
        return base_path


# Example usage:
# logging_instance = Logging("/logs", "dataset1", "experiment1", "modelX", ["config.yaml", "params.json"])


