#!/usr/bin/env python3
"""Bounded-cache Pick-a-Pic parquet streaming for distributed training."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import pathlib
import random
import re
import time
from dataclasses import dataclass
from typing import Iterator

import pyarrow.parquet as pq
import requests
import torch
from huggingface_hub import HfFolder, hf_hub_url


REQUIRED_COLUMNS = ("jpg_0", "jpg_1", "label_0", "caption")


@dataclass(frozen=True)
class Segment:
    filename: str
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return seed ^ int.from_bytes(digest[:8], "big")


class _BoundedShardCache:
    RANGE_CHUNK_BYTES = 32 * 1024 * 1024
    # Four DDP ranks download concurrently. Two range workers per rank keeps
    # enough network parallelism without opening 16 competing HTTP streams.
    RANGE_WORKERS = 2

    def __init__(
        self,
        repo_id: str,
        revision: str,
        local_data_dir: pathlib.Path,
        cache_dir: pathlib.Path,
        retries: int,
    ) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.local_data_dir = local_data_dir
        self.cache_dir = cache_dir
        self.retries = retries
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        token = os.environ.get("HF_TOKEN") or HfFolder.get_token()
        self.auth_headers = {"Authorization": f"Bearer {token}"} if token else {}

    def _request_size(self, url: str) -> tuple[int, bool]:
        for attempt in range(1, self.retries + 1):
            headers = dict(self.auth_headers)
            headers["Range"] = "bytes=0-0"
            try:
                with requests.get(
                    url,
                    headers=headers,
                    stream=True,
                    allow_redirects=True,
                    timeout=(30, 60),
                ) as response:
                    response.raise_for_status()
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes 0-0/(\d+)", content_range)
                    if response.status_code == 206 and match:
                        return int(match.group(1)), True
                    content_length = int(
                        response.headers.get("Content-Length", "0") or 0
                    )
                    if not content_length:
                        raise OSError("server did not report shard size")
                    return content_length, False
            except Exception as error:
                if attempt >= self.retries:
                    raise RuntimeError(
                        f"shard metadata request failed after {self.retries} attempts"
                    ) from error
                delay = min(2 ** (attempt - 1), 20)
                print(
                    f"metadata_retry={attempt}/{self.retries} "
                    f"error={type(error).__name__}:{error} sleep={delay}s",
                    flush=True,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    def _download_range(
        self,
        url: str,
        chunk_path: pathlib.Path,
        start: int,
        stop: int,
        filename: str,
    ) -> None:
        expected = stop - start
        if chunk_path.exists() and chunk_path.stat().st_size > expected:
            chunk_path.unlink()
        if chunk_path.exists() and chunk_path.stat().st_size == expected:
            return

        for attempt in range(1, self.retries + 1):
            present = chunk_path.stat().st_size if chunk_path.exists() else 0
            # A slow-read exception can be raised immediately after the final
            # bytes were written. Re-check on every retry so we do not send an
            # invalid Range such as bytes=33554432-33554431.
            if present == expected:
                return
            if present > expected:
                chunk_path.unlink()
                present = 0
            request_start = start + present
            headers = dict(self.auth_headers)
            headers["Range"] = f"bytes={request_start}-{stop - 1}"
            try:
                with requests.get(
                    url,
                    headers=headers,
                    stream=True,
                    allow_redirects=True,
                    timeout=(30, 60),
                ) as response:
                    response.raise_for_status()
                    match = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)",
                        response.headers.get("Content-Range", ""),
                    )
                    if response.status_code != 206 or not match:
                        raise OSError(
                            f"range request was not honored (status={response.status_code})"
                        )
                    returned_start, returned_stop = int(match.group(1)), int(match.group(2))
                    if returned_start != request_start or returned_stop != stop - 1:
                        raise OSError(
                            f"wrong content range {returned_start}-{returned_stop}, "
                            f"wanted {request_start}-{stop - 1}"
                        )

                    window_started = time.monotonic()
                    window_bytes = 0
                    with chunk_path.open("ab") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            window_bytes += len(chunk)
                            elapsed = time.monotonic() - window_started
                            if elapsed >= 60:
                                rate = window_bytes / elapsed
                                # A slow but progressing transfer is preferable
                                # to repeatedly discarding and reopening it.
                                if rate < 64 * 1024:
                                    raise OSError(
                                        f"slow range stream {rate / 1024:.0f} KiB/s"
                                    )
                                window_started = time.monotonic()
                                window_bytes = 0

                actual = chunk_path.stat().st_size
                if actual != expected:
                    raise OSError(
                        f"incomplete range: expected {expected} bytes, got {actual}"
                    )
                return
            except Exception as error:
                if attempt >= self.retries:
                    raise RuntimeError(
                        f"range download failed after {self.retries} attempts: "
                        f"{filename} [{start}:{stop}]"
                    ) from error
                delay = min(2 ** (attempt - 1), 20)
                print(
                    f"range_retry={attempt}/{self.retries} file={filename} "
                    f"range={start}:{stop} error={type(error).__name__}:{error} "
                    f"sleep={delay}s",
                    flush=True,
                )
                time.sleep(delay)

    def _download_parallel(
        self,
        url: str,
        target: pathlib.Path,
        filename: str,
        total_size: int,
    ) -> None:
        chunk_dir = target.with_suffix(target.suffix + ".chunks")
        chunk_dir.mkdir(parents=True, exist_ok=True)
        ranges = [
            (start, min(start + self.RANGE_CHUNK_BYTES, total_size))
            for start in range(0, total_size, self.RANGE_CHUNK_BYTES)
        ]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.RANGE_WORKERS
        ) as pool:
            futures = [
                pool.submit(
                    self._download_range,
                    url,
                    chunk_dir / f"{index:05d}.part",
                    start,
                    stop,
                    filename,
                )
                for index, (start, stop) in enumerate(ranges)
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        assembling = target.with_suffix(target.suffix + ".assembling")
        assembling.unlink(missing_ok=True)
        with assembling.open("wb") as output:
            for index, (start, stop) in enumerate(ranges):
                chunk_path = chunk_dir / f"{index:05d}.part"
                if chunk_path.stat().st_size != stop - start:
                    raise OSError(f"invalid cached range {chunk_path}")
                with chunk_path.open("rb") as source:
                    while block := source.read(4 * 1024 * 1024):
                        output.write(block)
                chunk_path.unlink()
        chunk_dir.rmdir()
        if assembling.stat().st_size != total_size:
            raise OSError(
                f"assembled shard has {assembling.stat().st_size} bytes, "
                f"expected {total_size}"
            )
        os.replace(assembling, target)

    def _download_sequential(
        self, url: str, target: pathlib.Path, filename: str, total_size: int
    ) -> None:
        part = target.with_suffix(target.suffix + ".part")
        for attempt in range(1, self.retries + 1):
            offset = part.stat().st_size if part.exists() else 0
            headers = dict(self.auth_headers)
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                with requests.get(
                    url,
                    headers=headers,
                    stream=True,
                    allow_redirects=True,
                    timeout=(30, 60),
                ) as response:
                    response.raise_for_status()
                    append = offset > 0 and response.status_code == 206
                    if offset and not append:
                        offset = 0
                    with part.open("ab" if append else "wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                actual = part.stat().st_size
                if actual != total_size:
                    raise OSError(
                        f"incomplete shard: expected {total_size} bytes, got {actual}"
                    )
                os.replace(part, target)
                return
            except Exception as error:
                if attempt >= self.retries:
                    raise RuntimeError(
                        f"sequential download failed after {self.retries} attempts: "
                        f"{filename}"
                    ) from error
                delay = min(2 ** (attempt - 1), 20)
                print(
                    f"stream_retry={attempt}/{self.retries} file={filename} "
                    f"error={type(error).__name__}:{error} sleep={delay}s",
                    flush=True,
                )
                time.sleep(delay)

    def obtain(self, filename: str) -> tuple[pathlib.Path, bool]:
        existing = self.local_data_dir / pathlib.Path(filename).name
        if existing.is_file() and not existing.is_symlink():
            return existing, False

        target = self.cache_dir / pathlib.Path(filename).name
        if target.is_file():
            return target, True
        url = hf_hub_url(
            self.repo_id,
            filename,
            repo_type="dataset",
            revision=self.revision,
        )
        total_size, supports_ranges = self._request_size(url)
        if supports_ranges:
            self._download_parallel(url, target, filename, total_size)
        else:
            print(f"stream_no_range file={filename}; using sequential download", flush=True)
            self._download_sequential(url, target, filename, total_size)
        pq.ParquetFile(target).metadata
        return target, True


class PickAPicStreamingDataset(torch.utils.data.IterableDataset):
    """Rank-partitioned iterable with deterministic shard/row shuffling.

    Every rank receives the same padded length, so DDP performs the same number
    of collectives. Existing materialized shards are reused; remote shards are
    prefetched one at a time and removed after consumption.
    """

    def __init__(
        self,
        manifest_path: str | pathlib.Path,
        local_data_dir: str | pathlib.Path,
        stream_cache_root: str | pathlib.Path,
        rank: int,
        world_size: int,
        seed: int,
        start_sample: int = 0,
        retries: int = 40,
        pad_to_multiple: int = 1,
        prefetch_next_shard: bool = False,
    ) -> None:
        super().__init__()
        manifest = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
        self.repo_id = str(manifest["repo_id"])
        self.revision = str(manifest.get("revision", "main"))
        self.target_rows = int(manifest["target_rows"])
        self.rows_per_shard = int(manifest.get("rows_per_shard", 0))
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.prefetch_next_shard = prefetch_next_shard

        files = list(manifest["files"])
        file_rows = manifest.get("file_rows")
        valid_row_indices = manifest.get("valid_row_indices")
        if valid_row_indices is not None:
            if not isinstance(valid_row_indices, dict):
                raise ValueError("streaming manifest valid_row_indices must be an object")
            missing = [filename for filename in files if filename not in valid_row_indices]
            if missing:
                raise ValueError(
                    f"streaming manifest is missing valid-row indices for {len(missing)} files"
                )
            for filename in files:
                indices = valid_row_indices[filename]
                if not isinstance(indices, list) or any(
                    not isinstance(index, int) or index < 0 for index in indices
                ):
                    raise ValueError(f"invalid valid_row_indices entry for {filename}")
                if isinstance(file_rows, dict) and int(file_rows[filename]) != len(indices):
                    raise ValueError(
                        f"row-count/index mismatch for {filename}: "
                        f"{file_rows[filename]} != {len(indices)}"
                    )
        self.valid_row_indices = valid_row_indices
        if file_rows is not None:
            if not isinstance(file_rows, dict):
                raise ValueError("streaming manifest file_rows must be an object")
            missing = [filename for filename in files if filename not in file_rows]
            if missing:
                raise ValueError(
                    f"streaming manifest is missing row counts for {len(missing)} files"
                )
        elif self.rows_per_shard <= 0:
            raise ValueError(
                "streaming manifest needs exact file_rows or a positive rows_per_shard"
            )
        random.Random(seed).shuffle(files)
        chunks: list[tuple[str, int]] = []
        remaining = self.target_rows
        for filename in files:
            if remaining <= 0:
                break
            available = (
                int(file_rows[filename])
                if file_rows is not None
                else self.rows_per_shard
            )
            if available <= 0:
                raise ValueError(f"invalid row count for {filename}: {available}")
            take = min(available, remaining)
            chunks.append((str(filename), take))
            remaining -= take
        if remaining:
            raise ValueError(f"streaming manifest is short by {remaining} rows")

        base, remainder = divmod(self.target_rows, world_size)
        raw_lengths = [base + (1 if index < remainder else 0) for index in range(world_size)]
        padded_length = math.ceil(max(raw_lengths) / pad_to_multiple) * pad_to_multiple
        rank_begin = sum(raw_lengths[:rank])
        rank_end = rank_begin + raw_lengths[rank]

        segments: list[Segment] = []
        cursor = 0
        for filename, count in chunks:
            chunk_end = cursor + count
            left = max(cursor, rank_begin)
            right = min(chunk_end, rank_end)
            if left < right:
                segments.append(Segment(filename, left - cursor, right - cursor))
            cursor = chunk_end
            if cursor >= rank_end:
                break

        if sum(segment.length for segment in segments) != raw_lengths[rank]:
            raise AssertionError("rank partition does not cover its assigned samples")
        if not 0 <= start_sample <= padded_length:
            raise ValueError(f"invalid streaming resume offset {start_sample}/{padded_length}")

        unique_skip = min(start_sample, raw_lengths[rank])
        trimmed: list[Segment] = []
        for segment in segments:
            if unique_skip >= segment.length:
                unique_skip -= segment.length
                continue
            if unique_skip:
                segment = Segment(segment.filename, segment.start + unique_skip, segment.stop)
                unique_skip = 0
            trimmed.append(segment)

        original_padding = padded_length - raw_lengths[rank]
        padding_skip = max(0, start_sample - raw_lengths[rank])
        self.padding = max(0, original_padding - padding_skip)
        self.segments = trimmed
        self.length = padded_length - start_sample
        self.cache = _BoundedShardCache(
            repo_id=self.repo_id,
            revision=self.revision,
            local_data_dir=pathlib.Path(local_data_dir),
            cache_dir=pathlib.Path(stream_cache_root) / f"rank{rank}",
            retries=retries,
        )
        print(
            f"stream_partition rank={rank}/{world_size} rows={self.length} "
            f"segments={len(self.segments)} padding={self.padding} start_sample={start_sample}",
            flush=True,
        )

    def __len__(self) -> int:
        return self.length

    def _rows(self, segment: Segment, path: pathlib.Path) -> Iterator[dict[str, object]]:
        parquet = pq.ParquetFile(path)
        available = parquet.metadata.num_rows
        eligible = (
            self.valid_row_indices[segment.filename]
            if self.valid_row_indices is not None
            else list(range(available))
        )
        if eligible and max(eligible) >= available:
            raise RuntimeError(
                f"{segment.filename} valid-row plan references rows outside {available}"
            )
        if segment.stop > len(eligible):
            raise RuntimeError(
                f"{segment.filename} only has {len(eligible)} eligible rows; "
                f"plan needs {segment.stop}"
            )
        table = parquet.read(columns=list(REQUIRED_COLUMNS))
        columns = table.to_pydict()
        order = list(eligible)
        random.Random(_stable_seed(self.seed, segment.filename)).shuffle(order)
        for logical_index in range(segment.start, segment.stop):
            row_index = order[logical_index]
            yield {name: columns[name][row_index] for name in REQUIRED_COLUMNS}

    def __iter__(self) -> Iterator[dict[str, object]]:
        if torch.utils.data.get_worker_info() is not None:
            raise RuntimeError("PickAPicStreamingDataset requires dataloader_num_workers=0")
        if not self.segments:
            return

        last_sample: dict[str, object] | None = None
        if self.prefetch_next_shard:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.cache.obtain, self.segments[0].filename)
                for index, segment in enumerate(self.segments):
                    path, ephemeral = future.result()
                    next_future = (
                        pool.submit(self.cache.obtain, self.segments[index + 1].filename)
                        if index + 1 < len(self.segments)
                        else None
                    )
                    try:
                        for sample in self._rows(segment, path):
                            last_sample = sample
                            yield sample
                    finally:
                        if ephemeral and path.is_file():
                            path.unlink()
                    if next_future is not None:
                        future = next_future
        else:
            # Keep at most one ephemeral shard per rank. This is important on
            # Hessian where /tmp is small and the model cache shares the disk.
            for segment in self.segments:
                path, ephemeral = self.cache.obtain(segment.filename)
                try:
                    for sample in self._rows(segment, path):
                        last_sample = sample
                        yield sample
                finally:
                    if ephemeral and path.is_file():
                        path.unlink()

        if self.padding:
            if last_sample is None:
                raise RuntimeError("cannot pad an empty streaming partition")
            for _ in range(self.padding):
                yield dict(last_sample)
