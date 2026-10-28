<!-- SPDX-License-Identifier: Apache-2.0 -->

# W7-A-001: A Parquet page header can still make `parquet` allocate up to 2 GiB before a short read

| Field | Value |
| :--- | :--- |
| Title | A Parquet page header can still make `parquet` allocate up to 2 GiB before a short read |
| Severity | low |
| Status | blocked-upstream |
| Kind | bug |
| Files | `crates/core/src/ingest.rs` (`decode_parquet_projected`, `VfsRangeReader`), `parquet-53.4.1/src/file/serialized_reader.rs:603-612` |

## Description

Found by reading `parquet` 53.4.1 while fixing W7-B-002 (not executed). Since W7-A, every column
chunk a load decodes is checked before decoding: it must lie inside the file and be at most
`memory_limit` bytes (`check_parquet_chunk_ranges`). A page header *inside* an honest-sized chunk
is still trusted by `parquet`'s synchronous page reader:

```rust
let data_len = header.compressed_page_size as usize;  // up to i32::MAX
*remaining -= data_len;                                // wraps in release if data_len > remaining
let mut buffer = Vec::with_capacity(data_len);         // infallible
let read = read.take(data_len as u64).read_to_end(&mut buffer)?;
```

So a small file whose page header claims `compressed_page_size = i32::MAX` makes `parquet`
reserve about 2 GiB before the short read is detected. On a host without memory overcommit
(Windows commit limit, `vm.overcommit_memory = 2`) a failed reservation aborts the process, which
`contain_decoder_panic` cannot catch. The extension cannot intercept the allocation: it happens
inside `parquet` before the `ChunkReader` sees a read. A prefetched chunk (W6-A-001) makes the
following read short, so the load fails, but only after the reservation. Local loads are
affected the same way.

## Suggested fix

Upstream: `parquet` should check `compressed_page_size` against the column chunk's remaining
bytes before allocating; check whether a later `parquet` release does when the workspace moves
past 53, and report it upstream if not. In the extension, the only option short of that would be to
parse each page header with `parquet`'s thrift types before handing the chunk to the decoder,
which duplicates the reader. Test once upstream checks it: a file with one page header patched to
`compressed_page_size = i32::MAX` is an error and the peak allocation stays below the chunk size.
