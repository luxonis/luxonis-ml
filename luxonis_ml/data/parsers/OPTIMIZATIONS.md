# Parser performance notes

Notes on optimizations in `luxonis_ml/data/parsers` that are not obvious from
reading the code, so that nobody has to rediscover why a parser is shaped the
way it is — or undoes one by "simplifying" it.

Every entry here is measured by the benchmark suite and guarded by a regression
test. The *why* of each individual change lives in a comment at the site; this
file is the map, the numbers, and what is deliberately still slow.

## Re-measuring

```bash
pytest -m benchmark tests/test_data/parsers/benchmarks --benchmark-json before.json
# ... change a parser ...
pytest -m benchmark tests/test_data/parsers/benchmarks --benchmark-compare before.json
```

Add `--benchmark-repeat` and a bigger `--benchmark-scale` before believing a
difference of a few percent; the `± %` column says how much of one is scatter.

Each parser is fed a generated dataset covering every feature it supports. See
[CONTRIBUTING.md](../../../CONTRIBUTING.md#parser-benchmarks) for the options.

## Proving a change is output-preserving

Speed is not worth a parser that behaves differently, so optimizations are
gated on an exact-output comparison rather than on the test suite alone:

```bash
python tools/profile_parsers.py digests before.json   # before the change
python tools/profile_parsers.py compare before.json   # after it
```

`compare` regenerates each benchmark dataset and checks the ordered file list,
the splits, the skeleton metadata, the reported parse issues and a hash of
every record, in order. It printed `identical` for all 18 dataset types for
every change described here.

The generated datasets are built at a fixed path rather than in a temporary
directory: a parser that downloads a referenced image names it after a hash of
the URL, so a moving dataset root would make two runs of identical code differ.

## Measured effect

Default benchmark scale, 2000 images per split, single process. "round 1"
optimized the parsers under the old contract; "round 2" is the streaming API.

| dataset type                              | original | round 1 | round 2 | total |      peak MiB |
| ----------------------------------------- | -------: | ------: | ------: | ----: | ------------: |
| `clsdir`                                  |   0.426s |  0.046s |  0.023s | 18.4x |    2.7 -> 0.1 |
| `segmask`                                 |   2.170s |  0.189s |  0.201s | 10.8x |    1.5 -> 0.7 |
| `tfcsv`                                   |   3.308s |  0.410s |  0.379s |  8.7x |   28.8 -> 8.3 |
| `native`                                  |   8.167s |  1.329s |  1.085s |  7.5x | 144.5 -> 50.0 |
| `yolov8instancesegmentation`              |   7.670s |  0.908s |  1.028s |  7.5x |    4.4 -> 1.6 |
| `fiftyone-classification`                 |   0.115s |  0.034s |  0.024s |  4.8x |    5.0 -> 1.6 |
| `yolov4`                                  |   2.371s |  0.561s |  0.658s |  3.6x |    3.6 -> 1.8 |
| `darknet`                                 |   0.725s |  0.230s |  0.211s |  3.4x |    3.0 -> 2.5 |
| `yolov6`                                  |   0.763s |  0.241s |  0.223s |  3.4x |    7.9 -> 2.4 |
| `yolov8keypoints`                         |   2.119s |  0.791s |  0.678s |  3.1x |    3.0 -> 1.0 |
| `coco`                                    |   3.868s |  1.509s |  1.246s |  3.1x |  71.2 -> 70.6 |
| `yolov8`                                  |   0.826s |  0.414s |  0.391s |  2.1x |    2.9 -> 6.0 |
| `solo`                                    |   6.754s |  6.212s |  3.804s |  1.8x |    5.7 -> 7.1 |
| `createml`                                |   1.321s |  0.815s |  0.758s |  1.7x |  32.7 -> 14.2 |
| `voc`                                     |   1.856s |  1.205s |  1.123s |  1.7x |   19.8 -> 1.5 |
| `ultralytics-ndjson`                      |   1.995s |  1.706s |  1.421s |  1.4x |    3.8 -> 0.1 |
| `ultralytics-ndjson-instancesegmentation` |   1.223s |  0.983s |  0.888s |  1.4x |    3.9 -> 0.1 |
| `ultralytics-ndjson-keypoints`            |   1.022s |  1.014s |  0.835s |  1.2x |    3.8 -> 0.1 |

46.7s -> 15.0s over the whole suite; 348 MiB -> 170 MiB of peak allocation.

Two caveats on reading the round-2 column. The benchmark times the parser
only: building the file list used to be the parser's work and was timed,
and is now the importer's, so the two rounds do not measure exactly the
same span. And `parse()` is 4-7% of a real `import_dataset` — the rest is
record validation, parquet writing and progress rendering — so these
numbers are not end-to-end import numbers.

## What the parser API looks like now

A parser walks its source once. `ParseResult` carries an iterator of
`(split, record)` pairs and the skeleton metadata; the importer collects
each split's files as the records stream past, because the only thing that
needed a file list — `make_splits` — runs after every record is added.

`detect()` returns the `Layout` it discovered and the importer hands that
back to `parse()`, so a source is inspected once rather than once by
`supports()` and again by `parse()`.

`enumerate_files()` exists for the one feature that genuinely needs the
files up front, count-based `split_ratios`. A parser that cannot answer
cheaply returns `None` and the importer falls back to a throwaway parse —
for that case alone, not for every import.

## The two systemic ones

### Every parser parsed its source twice

The original `ParsedDataset` required `files` to be complete before `records`
was consumed, so that a dataset importer could pick a count-based subset
without touching the records. Every parser satisfied that by building a
second, throwaway generator:

```python
added_images = self._get_added_images(generator())   # the old contract
return ParsedDataset(generator(), {}, added_images)
```

That runs the entire parse twice and discards everything but the file paths —
including, for YOLOv8 segmentation, decoding every image a second time. It also
rebuilt a `Path` per *record* rather than per file, which made
`pathlib.parse_parts` the hottest function of a YOLOv8 parse at 27%.

Each parser now derives `files` directly from the same source of truth its
generator walks: the image listing, the annotation index, the JSON image table.
Two of them need slightly more care, and both say so at the site:

- **YOLOv8** yields no record at all for an image whose label lines are *all*
  malformed, so the file list is the image listing filtered by a scan of the
  label file that stops at the first usable line.
- **SOLO** cannot cheaply predict which captures yield records, so it does not
  answer: its `_split_files` returns `None` and the importer falls back to the
  throwaway parse described above — for count-based splits only.

Prefer this shape in a new parser. A parser that genuinely cannot derive its
file list any other way returns `None` and pays for that fallback.

### Work that depends only on the image was done per annotation

The YOLOv8 segmentation branch called `cv2.imread` once per polygon to read the
image size, so an image with 10 polygons was decoded 10 times — and then all of
that happened twice because of the second pass. 20 decodes became 1. That one
change is most of the 7.5x.

The same pattern, at smaller scale, was in path resolution: `resolve()` is a
`realpath` walk that lstats every component, and parsers called it per
annotation on directories that do not change during a split.

## Eager failure, and what the streaming API gave up

Removing a pass removes the errors it used to raise. Where the discarded
pass was the only thing that read a file, an unreadable file stopped
failing at parse time and started failing part-way through `dataset.add`
— which leaves a registered, half-populated dataset behind.

`SegmentationMaskDirectoryParser` and `YOLOv8Parser` still check
`cv2.haveImageReader` while listing a split's files, so an undecodable
image fails the import up front at the cost of a header read rather than
a decode.

**`SOLOParser` no longer can.** Its check lived in the file-enumeration
pass, and the streaming API has no such pass: an undecodable mask is now
found while the records stream, from inside `dataset.add`. This is the one
place where one-pass parsing and fail-before-you-write genuinely conflict,
and the parser cannot resolve it alone. The guarantee is restored one
level up instead: `BaseDataset.import_dataset` deletes any dataset it
created when the import fails, so a mid-stream SOLO failure does not
leave a half-populated dataset behind.

## Per-parser notes

Only the nontrivial ones; each has a comment at the site and a regression test.

| Parser                    | Optimization                                                                                                                                                                                                                                                                                                                                                               |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coco`                    | One pass over the annotation index; annotation files decoded once per parse instead of up to 13 times; `clean_annotations` writes JSON child by child, which reaches the C encoder that `json.dump`'s chunked API silently bypasses (245ms -> 69ms) without buffering the whole document; the taken-instance-id set is built only when an annotation actually lacks an id. |
| `yolov8`                  | One decode per image instead of per polygon; array-free polygon bounding box and keypoint grouping, keeping the numpy path for shapes the fast one cannot express.                                                                                                                                                                                                         |
| `solo`                    | File enumeration walks the split without building record payloads.                                                                                                                                                                                                                                                                                                         |
| `voc`                     | File list from the parsed annotations; scalar box normalization instead of a numpy array per box; records stream from a compact per-image index.                                                                                                                                                                                                                           |
| `tfcsv`                   | Each filename spelling resolved at most once; one realpath walk for the split directory instead of one per image; CSV rows streamed as tuples.                                                                                                                                                                                                                             |
| `native`                  | One pass over the annotation document; each distinct media path resolved once per split.                                                                                                                                                                                                                                                                                   |
| `segmask`                 | One pass; image lookup from a single directory listing; per-class masks without a full sort or a zero fill.                                                                                                                                                                                                                                                                |
| `clsdir`                  | One walk feeds both the file list and the records; each class directory resolved once instead of each image.                                                                                                                                                                                                                                                               |
| `createml`                | File list falls out of the annotation loop; separator-free references skip generic manifest resolution.                                                                                                                                                                                                                                                                    |
| `yolov4`, `yolov6`        | File list from the image listing; one image-directory listing per parse; directory images matched by name before falling back to resolving.                                                                                                                                                                                                                                |
| `darknet`                 | File list from the image listing; split listing reused from `validate_split`.                                                                                                                                                                                                                                                                                              |
| `fiftyone-classification` | File list resolved from the label map; `labels.json` parsed once per split.                                                                                                                                                                                                                                                                                                |
| `ultralytics-ndjson`      | One reduction per axis when fitting a polygon box; two-column keypoint layouts no longer build a visibility column only to discard it.                                                                                                                                                                                                                                     |

## Still on the table

Measured, deliberately not done, in rough order of value. All of them live in
shared code, where a mistake costs every parser at once. The double split
discovery that used to head this list is gone: `detect()` returns the `Layout`
and `parse()` is handed it.

Note that `parse()` itself is only 4-7% of an `import_dataset`. The larger
remaining costs are on the ingestion side - the progress bar redrawing per
record, and pydantic validation - and are out of scope for these notes.

1. **`resolve_manifest_path` builds four or five path objects per call.** It
   constructs `PureWindowsPath(raw)` and `Path(raw)`, then `parse_manifest_path`
   repeats both. A fast path for a relative POSIX string with no backslash would
   skip almost all of it, for every parser that reads a manifest. Note that
   `base_dir.resolve() / name` is *not* equivalent — `resolve()` also follows a
   symlinked final component.
1. **`_list_images` rebuilds its 20-element suffix set on every call** and
   builds a `Path` per directory entry before filtering. Hoisting the set is
   free. Rewriting it on `os.scandir` was measured and is *not* faster on
   Python 3.10 (2.07ms vs 1.77ms for 800 files): `Path.glob` builds children
   through an internal shortcut that a scandir version has to pay for.
1. **`validate_split` only needs to know whether one image exists**, but
   `_list_images` materializes the whole listing.
1. **SOLO still reads every frame JSON twice**, once to pair captures with
   their masks and once to build the records. This is the last place where the
   record contract itself is the cost; letting a parser publish its files
   incrementally would remove it.
