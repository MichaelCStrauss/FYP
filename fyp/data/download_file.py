import requests
from loguru import logger
from tqdm import tqdm

def download_file(source: str, destination: str):
    with open(destination, "wb") as f:
        logger.info(f"Downloading {source} to {destination}")
        response = requests.get(source, stream=True)
        total_length = int(response.headers.get('content-length')) / 1_000_000

        if total_length is None: # no content length header
            f.write(response.content)
        else:
            with tqdm(total=total_length) as p:
                for data in response.iter_content(chunk_size=1_000_000):
                    p.update(len(data) / 1_000_000)
                    f.write(data)