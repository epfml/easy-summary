import os

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib
import PIL.Image

from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent


USER_AGENT = get_datasets_user_agent()

def fetch_single_image(image_url, timeout=None, retries=0):
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
            break
        except Exception:
            image = None
    return image

def fetch_images(batch, num_threads, timeout=None, retries=0, image_dir=None):
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        images = list(executor.map(fetch_single_image_with_args, batch["image_url"]))
        # "http://static.f11ckY.com/2723/4385058960_b0f291553e.jpg" would be saved at /share/project/wangzekun/datasets/sbu_captions/hugface/2723/4385058960_b0f291553e.jpg.
        dirnames = [os.path.join(image_dir, url.split("/")[-2]) for url in batch["image_url"]]
        file_names = [url.split("/")[-1] for url in batch["image_url"]]

    for i, image in enumerate(images):
        if image is not None:
            # save images
            if not os.path.exists(dirnames[i]):
                os.makedirs(dirnames[i])
            image.save(os.path.join(dirnames[i], file_names[i]))

    batch["image"] = images
    return batch

num_threads = 20
timeout = 5
retries = 5
image_dir = "sub_captions/hugface"

dset = load_dataset("sbu_captions", cache_dir=image_dir)
dset = dset.map(partial(fetch_images, num_threads=num_threads, timeout=timeout, retries=retries, image_dir=image_dir), batched=True, batch_size=100)