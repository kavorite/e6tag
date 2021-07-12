import csv
import gzip
import io
import os
import urllib.request as http
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import BoundedSemaphore
from urllib.error import HTTPError

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tqdm import tqdm

SHARD_LENGTH = 256
SHARD_ROOT = "D:/yiff"  # change at your discretion


def download_with_progress(endpoint, output_path=None, chunk_size=4096):
    output_path = output_path or os.path.basename(endpoint)
    with tqdm(unit_scale=True, unit_divisor=1024, unit="B") as progress:
        with open(output_path, "wb+") as ostrm:
            while (progress.n or -1) < (progress.total or 0):
                try:
                    headers = {"User-Agent": "e6tag hydration by kavorite"}
                    if progress.total:
                        headers["Range"] = f"bytes={progress.n or 0}-{progress.total-1}"
                    request = http.Request(endpoint, headers=headers)
                    with http.urlopen(request, timeout=2) as rsp:
                        progress.total = progress.total or int(
                            rsp.headers["Content-Length"]
                        )
                        progress.refresh()
                        while chunk := rsp.read(chunk_size):
                            progress.update(len(chunk))
                            ostrm.write(chunk)
                except OSError:
                    continue


def fetch_metadata():
    yesterday = (datetime.today() - timedelta(days=1)).date()
    if (
        not os.path.exists("tags.csv.gz")
        or datetime.fromtimestamp(os.stat("tags.csv.gz").st_mtime).date() <= yesterday
    ):
        endpoint = f"https://e621.net/db_export/tags-{yesterday.isoformat()}.csv.gz"
        download_with_progress(endpoint, "tags.csv.gz")

    if (
        not os.path.exists("posts.csv.gz")
        or datetime.fromtimestamp(os.stat("posts.csv.gz").st_mtime).date() <= yesterday
    ):
        endpoint = f"https://e621.net/db_export/posts-{yesterday.isoformat()}.csv.gz"
        download_with_progress(endpoint, "posts.csv.gz")


def top_tags(k=-1):
    with gzip.open("./tags.csv.gz") as istrm:
        tag_rows = csv.DictReader(io.TextIOWrapper(istrm, encoding="utf8"))
        tags = [
            tag
            for tag in tqdm(tag_rows)
            if tag["name"].isprintable()
            and tag["category"] in "01"  # allow general and species tags
            and "comic" not in tag["name"]
        ]

    tags.sort(key=lambda tag: int(tag["post_count"]))
    tags = tags[::-1][:k]
    return {tag["name"]: i for i, tag in enumerate(tags)}


def tags_of(post, tag_idx):
    post_tags = [t for t in post["tag_string"].split() if t in tag_idx]
    post_tags.sort(key=lambda t: tag_idx[t])
    post_tags = post_tags[::-1]
    return post_tags


def bytes_feature(x):
    x = tf.train.BytesList(value=[x])
    x = tf.train.Feature(bytes_list=x)
    return x


def int64_feature(x):
    x = tf.train.Int64List(value=x)
    x = tf.train.Feature(int64_list=x)
    return x


def make_example(post, image_str):
    def sparse_row(A, i):
        return tf.squeeze(tf.gather(A.indices, tf.where(A.indices[:, 0] == i)))[:, 1]

    tag_indxs = sparse_row(labels, int(post["id"])).numpy()
    tag_names_ft = bytes_feature(b" ".join(enc_tag_names[tag_indxs]))
    tag_indxs_ft = int64_feature(tag_indxs)
    post_id_ft = int64_feature([post_id])
    image_str_ft = bytes_feature(image_str)
    feature = dict(
        image_str=image_str_ft,
        tag_indxs=tag_indxs_ft,
        tag_names=tag_names_ft,
        post_id=post_id_ft,
    )
    return tf.train.Example(features=tf.train.Features(feature=feature))


def download_post(post, sample=True):
    try:
        endpoint = post["link"]
        if not sample:
            endpoint = endpoint.replace("/sample/", "/")
        max_retries = 2
        for _ in range(max_retries + 1):
            try:
                with http.urlopen(endpoint) as rsp:
                    return rsp.read()
            except OSError:
                continue
        return None
    except HTTPError as err:
        if err.status == 404:
            return None
        else:
            raise err


def download_posts(posts, sample=False):
    concurrency = os.cpu_count() * 4
    posts = iter(posts)
    semaphore = BoundedSemaphore(concurrency)
    with ThreadPoolExecutor() as pool:
        jobs = deque()
        while True:
            if semaphore.acquire(blocking=False):
                post = next(posts, None)
                if post is not None:
                    job = pool.submit(download_post, post, sample=sample)
                    job.add_done_callback(lambda _: semaphore.release())
                    jobs.append(job)
                else:
                    semaphore.release()
            while jobs and jobs[0].done():
                yield jobs.popleft().result()


if __name__ == "__main__":
    csv.field_size_limit(1 << 20)
    posts_by_id = dict()
    fetch_metadata()
    tag_idx = top_tags(k=4096)
    hit_tags = {t: 64 for t in tag_idx}
    all_tags = set(tag_idx.keys())
    min_tags = 16

    with gzip.open("./posts.csv.gz") as istrm:
        post_rows = csv.DictReader(io.TextIOWrapper(istrm, encoding="utf8"))
        skip = 1_000_000
        print(f"skipping the first {skip} posts...\n")
        for i, post in tqdm(enumerate(post_rows), total=skip):
            if i + 1 >= skip:
                break
        goal = sum(hit_tags.values())
        print(f"pulling {goal} positive hits...")
        for post in tqdm(post_rows, total=2_800_000 - skip):
            if post["is_deleted"] == "t":
                continue
            if post["file_ext"] in {"webm", "gif", "swf"}:
                continue
            if "comic" in post["tag_string"]:
                continue
            post_tags = tags_of(post, tag_idx)
            if len(post_tags) < min_tags:
                continue
            post_id = int(post["id"])
            for t in post_tags:
                if t not in hit_tags:
                    continue
                hit_tags[t] -= 1
                if hit_tags[t] == 0:
                    del hit_tags[t]
                md5 = post["md5"]
                ext = post["file_ext"] or "jpg"
                link = f"https://static1.e621.net/data/sample/{md5[0:2]}/{md5[2:4]}/{md5}.{ext}"
                post["link"] = link
                post = {k: post[k] for k in ["link", "id", "tag_string"]}
                posts_by_id[post_id] = post
                break

    indices = set()
    for i, post in enumerate(tqdm(posts_by_id.values())):
        for t in tags_of(post, tag_idx):
            if t in tag_idx:
                j = tag_idx[t]
                indices.add((int(post["id"]), j))
    label_shape = (len(posts_by_id), len(tag_idx))
    label_values = np.ones(len(indices))
    labels = tf.SparseTensor(list(indices), label_values, label_shape)
    labels = tf.sparse.reorder(labels)

    enc_tag_names = [t.encode("utf8") for t in tag_idx]
    enc_tag_names = np.array(enc_tag_names, dtype=object)

    total_shards = int(tf.math.ceil(len(posts_by_id) / SHARD_LENGTH))
    zpad = int(tf.math.ceil(tf.math.log(float(total_shards)) / tf.math.log(10.0)) + 1)

    if not os.path.exists(SHARD_ROOT):
        os.makedirs(SHARD_ROOT)

    with open(os.path.join(SHARD_ROOT, "tags.txt"), "w+", encoding="utf8") as ostrm:
        ostrm.write("\n".join(tag_idx.keys()))

    all_posts = list(posts_by_id.values())
    print("sharding posts...")
    shard_posts = [[] for _ in range(total_shards)]
    for i, post in tqdm(enumerate(all_posts), total=len(all_posts)):
        shard_posts[i % total_shards].append(post)
    with tqdm(total=len(posts_by_id)) as progress:
        for i in range(total_shards):
            index = str(i).zfill(zpad)
            name = f"samples.shard{index}of{total_shards}.tfrecords"
            name = os.path.join(SHARD_ROOT, name)
            posts = shard_posts[i]
            if (
                os.path.exists(name)
                and i != total_shards - 1
                and os.stat(name).st_size > 200e6
            ):
                progress.update(len(posts))
                continue
            with tf.io.TFRecordWriter(name) as records:
                image_strs = download_posts(iter(posts))
                for post, image_str in zip(posts, image_strs):
                    progress.update()
                    if image_str is None:
                        continue
                    example = make_example(post, image_str)
                    records.write(example.SerializeToString())
