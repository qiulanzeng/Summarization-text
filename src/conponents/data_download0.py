import os
import urllib.request as request
import zipfile
from src.logging import logger
from src.utils.common import get_size
from pathlib import Path



class DataDownload:
    def __init__(self, config):
        self.config = config


    
    def download_file(self):
        os.makedirs(self.config['data_input']['dir'], exist_ok=True)
        if not os.path.exists(self.config['data_input']['local_data_file']):
            filename, headers = request.urlretrieve(
                url = self.config['data_input']['source_URL'],
                filename = self.config['data_input']['local_data_file']
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config['data_input']['local_data_file']))}")  

        
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config["data_input"]["dir"]
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config["data_input"]["local_data_file"], 'r') as zip_ref:
            zip_ref.extractall(unzip_path)