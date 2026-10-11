<div align="center">

# ParkingScan

**Parking occupancy from a single oblique photo of a car park —
a YOLO26s detector with SAHI tiling finds the cars, the slot grid is built from the painted
markings in the same image, and nobody draws slot polygons by hand.**

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688.svg)](https://fastapi.tiangolo.com/)
[![Ultralytics](https://img.shields.io/badge/Detector-YOLO26s-4B8BBE.svg)](https://docs.ultralytics.com/)
[![SAHI](https://img.shields.io/badge/SAHI-sliced%20inference-FF6F00.svg)](https://github.com/obss/sahi)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.x-5C3EE8.svg)](https://opencv.org/)
[![Docker](https://img.shields.io/badge/Docker-CPU%20image-2496ED.svg)](docker/Dockerfile)
[![Dataset](https://img.shields.io/badge/Kaggle-competition-20BEFF.svg)](https://www.kaggle.com/competitions/cars-object-detection)

</div>

---

ParkingScan is an end-to-end computer-vision service for car parks photographed at an angle with a
wide-angle camera: **one 4000×3000 frame in, occupancy JSON or an annotated image out**. It detects
cars, reads the **painted slot markings** to build the slot grid, and decides which slots are taken.
There is no per-site annotation step and no top-down rectification — the quadrilateral between two
neighbouring dividers already is a slot. Where the markings cannot be read, the service says so
and falls back to counting cars instead of inventing a grid.

> **Where the data comes from.** The model was trained on frames of the Kaggle competition
> [Cars object detection](https://www.kaggle.com/competitions/cars-object-detection). The original
> frames are not in this repository; every image you see here is derived from them — see
> [Dataset and image sources](#dataset-and-image-sources).
>
> **What is mine.** The service, the slot-grid algorithm, the tests and the results below.

![Occupancy on a clean frame](docs/screenshots/occupancy.jpg)

*Occupied slots are red, free slots are green, cars that match no slot are yellow, the cyan outline is
the working zone. Summary: 3 of 6 confirmed slots occupied.*

![Count-only mode](docs/screenshots/count_only.jpg)

*On this frame the markings cannot be read, so the system refuses to build a slot grid and reports
only the number of cars in the working zone (`calibration_status: failed`, `occupancy_rate: null`).
It does not invent slots it cannot see.*

Both screenshots are derived from dataset frames, see
[Dataset and image sources](#dataset-and-image-sources).

## Status

| Part | State |
|---|---|
| Car detection (YOLO26s + SAHI) | works |
| Slot grid from painted markings | works, with the limits described below |
| Occupancy, REST API, CLI | works |
| CPU Docker image | works, built and run with Docker Desktop on Windows |
| GPU in Docker | **not tested, no GPU image** |
| Database of snapshots | **planned, not in the code yet** |
| Number plate recognition (ANPR) | **planned, not in the code yet** |

## Features

### Slot grid from painted markings, one photo
- **Detection** — a YOLO26s model finds cars. The frame is resized to the training scale and processed
  in 640 px tiles with SAHI, because cars in these frames differ in size up to 25 times.
- **Local scale** — a robust (Huber) regression over the detected boxes gives the expected car size at
  every point of the frame. All thresholds are fractions of that size, not pixels.
- **Working zone** — the far field, where markings cannot be read, and the fisheye periphery are cut off.
- **Markings** — a paint mask (white top-hat with a local threshold) and a line segment detector give
  the slot dividers. Only the upper part of each car box is masked out, so dividers next to a car survive.
- **Rows and slots** — parallel dividers with a regular step form a row; the quadrilateral between two
  neighbouring dividers is a slot.

### Markings plus cars — a hybrid
On an oblique shot the markings are visible where slots are empty and hidden by car bodies where
slots are taken. So the visible dividers give a row its **direction and step**, and the cars give
its **extent**: the row is continued cell by cell with the measured step, and a cell is accepted
only if a car stands in it or an observed divider confirms its far border.

Every slot carries a `source`:

| Source | Meaning | Confidence |
|---|---|:---:|
| `observed` | both dividers seen in the image | up to **1.0** |
| `interpolated` | a divider restored inside the visible part of the row | **0.6** / **0.4** |
| `projected` | continued by the grid step, a car stands in it | **0.7** |
| `extrapolated` | continued by the grid step, confirmed by nothing | **0.4** |

### An occupancy rate that does not flatter itself
- **Ground point, not box overlap** — a slot is occupied when the middle of the bottom edge of a
  detection, raised by 15% of the box height, lies inside the slot polygon. On an oblique view the
  car body projects upwards and its box covers the neighbouring slot too.
- **One detection, one slot** — on a border between two polygons the slot in which the point lies
  deeper wins.
- **`occupancy_rate` is computed over confirmed slots only.** An `extrapolated` slot is confirmed by
  nothing; counting it as free would inflate the number of free slots and lower the rate. Unconfirmed
  slots are reported separately and count neither as free nor as occupied.

### A service that knows its limits
- **`ok` / `partial` / `failed`** — every response carries the calibration status and the reason.
- **Count-only mode** — when the grid cannot be built the response still returns the number of cars in
  the working zone, with `occupancy_rate: null`.
- **Cars without a slot are not hidden** — they are returned as `unassigned_detections` and drawn yellow.

### API, CLI and Docker
- **REST API** on FastAPI with typed responses and interactive docs at `/docs`.
- **CLI** for an image or a folder, printing the same JSON the API returns.
- **Docker** — a multi-stage CPU image, the service runs as an unprivileged user.

## Dataset and image sources

Source: **[Cars object detection — Kaggle competition](https://www.kaggle.com/competitions/cars-object-detection)**

The model was trained on **234 oblique 4000×3000 frames** of car parks shot with a wide-angle action
camera, **9 294 labelled cars**, about 40 cars per frame.

This is a competition, not a freely licensed dataset: the terms of use of the data are set by the
competition's [Rules](https://www.kaggle.com/competitions/cars-object-detection/rules) page. That is
why the original frames are not included in this repository.

**All images in this repository are derived from frames of that dataset**: the screenshots of the
system at work in `docs/screenshots/`, the training curves and validation mosaics in `training/`, and
the outputs embedded in `notebooks/CarsObjectDetection.ipynb`. The rights to the original images
belong to the dataset author.

| In the repository | What it is | Original frames needed |
|---|---|---|
| `models/cars_yolo26s_best.pt` | trained weights, 20 MB | no |
| `data/samples/ground_truth.json` | reference car boxes for 4 validation frames | yes, for the check scripts |
| `data/samples/slots_gt.json` | reference ground points for the slot-grid recall | yes, for the check scripts |
| `docs/screenshots/`, `training/`, notebook outputs | derived images | no |

The service and the tests run without the dataset. The two manual check scripts expect four frames
next to the reference files. Download the competition data and copy these files from
`train/train/images/` into `data/samples/`:

```
57441ed9-GOPR6559.JPG
ec5a75f7-GOPR6537.JPG
90a19daa-GOPR6757.JPG
f92cc929-GOPR6604.JPG
```

## Tech stack

| Area | Technology |
|---|---|
| Language | Python 3.14 |
| Detection | [Ultralytics](https://docs.ultralytics.com/) YOLO26s · [SAHI](https://github.com/obss/sahi) sliced inference · PyTorch |
| Markings and geometry | [OpenCV](https://opencv.org/) — top-hat, line segment detector, polygon tests |
| Local scale model | [scikit-learn](https://scikit-learn.org/) `HuberRegressor` · [NumPy](https://numpy.org/) |
| API | [FastAPI](https://fastapi.tiangolo.com/) · uvicorn · pydantic |
| Packaging | [uv](https://docs.astral.sh/uv/) · Docker, Docker Compose |
| Tests | pytest, synthetic frames |

## Results (real runs)

**Detector** — 35 validation images of the training notebook, which the model has not seen, at the
service threshold `conf = 0.25`:

| Metric | Value |
|---|:---:|
| F1, whole frame | **0.836** |
| F1, cars larger than 150 px | **0.892** |
| Median absolute error of the car count per image | **7.7 %** |

**Slot grid** — share of parked cars inside the working zone for which a slot was found, on the two
reference frames:

| Frame | Slot recall | History |
|---|:---:|---|
| `GOPR6537` | **0.50** (9 of 18) | 0.06 with markings only → 0.22 hybrid → 0.50 with the lower part of car boxes kept |
| `GOPR6757` | **0.43** (3 of 7) | 0.14 with markings only → 0.43 hybrid |

**Speed** — one 4000×3000 frame, detection, grid and occupancy together:

| Where | Time per frame |
|---|:---:|
| RTX 4070 Ti SUPER | **≈ 0.85 s** (detection alone ≈ 0.1 s) |
| CPU, inside the Docker image | **≈ 1.0 s** |

**Why there are two F1 values.** The dataset annotation of the far field is incomplete and
inconsistent: on some frames distant cars are not labelled, so correct detections are counted as
errors; on others they are labelled and the model does miss them. The whole-frame figure is therefore
both lowered and noisy. Cars larger than 150 px lie in the zone where the annotation can be trusted,
and 0.892 is the honest figure for the area the system actually analyses.

**The honest caveat.** Slot recall is 0.50 and 0.43, not close to 1. The acceptance threshold was
first set to 0.6 before looking at the data and was then replaced by the figures actually reached:
the limit is physical, not a tuning problem — the markings under parked cars are not visible. Read
the limitations below before comparing these numbers with systems that use a top-down camera or
hand-drawn slots.

## Installation

### Docker

The model weights (`models/cars_yolo26s_best.pt`) and the inference contract (`models/config.json`)
are part of the repository and are copied into the image. To use other weights, put them into
`models/` together with a matching `config.json` before building.

```bash
# 1. Clone
git clone https://github.com/TyomPapiyan/parkingscan.git
cd parkingscan

# 2. Build the CPU image (about 2.6 GB)
docker build -f docker/Dockerfile -t parkingscan .

# 3. Run
docker run --rm -p 8000:8000 -v "$(pwd)/data:/app/data" parkingscan
```

Or with Compose:

```bash
docker compose -f docker/docker-compose.yml up
```

> **Linux host — not tested.** The service runs inside the container as an unprivileged user with
> uid 1000 and writes site calibrations to `/app/data`. On a Linux host the mounted `data` directory
> therefore has to be writable by uid 1000. This case has not been tested; the image was checked with
> Docker Desktop on Windows.
>
> **GPU in Docker — not tested.** The image is built for CPU only, so there is no GPU image in this
> repository. The steps it would take: install the NVIDIA Container Toolkit on the host; in
> `docker/Dockerfile` install `torch` and `torchvision` from the CUDA index
> `https://download.pytorch.org/whl/cu130` instead of the CPU index; run the container with
> `--gpus all`. The service picks `cuda:0` automatically when it is available, and `/health` shows
> the device in use.

### Without Docker

```bash
# 1. Clone
git clone https://github.com/TyomPapiyan/parkingscan.git
cd parkingscan

# 2. Install dependencies into .venv
uv sync

# 3. Run
uv run uvicorn parkingscan.api.app:app --port 8000
```

> **Python note:** requires [uv](https://docs.astral.sh/uv/). The project is developed and tested on
> Python 3.14; `pyproject.toml` declares `>=3.12`, but older versions have not been tested.
>
> **GPU note:** on Windows and Linux `uv sync` installs PyTorch built for CUDA 13.0, and the service
> uses the GPU when one is available; otherwise it runs on CPU. This has been tested on Windows only.

## Usage

```bash
curl http://localhost:8000/health
curl -F "file=@lot.jpg" http://localhost:8000/occupancy/detect
curl -F "file=@lot.jpg" http://localhost:8000/occupancy/render -o result.jpg
```

Without the API, for one image or a folder:

```bash
uv run parkingscan lot.jpg
uv run parkingscan photos/ --out rendered/
```

The CLI prints one JSON line per image, the same object that `/occupancy/detect` returns, and with
`--out` saves the rendered images.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Status, device, version |
| POST | `/occupancy/detect` | Image, optional `site_id` and `conf` → occupancy JSON |
| POST | `/occupancy/render` | Same input → image with the overlay (`format`, `max_side`) |
| POST | `/sites/{site_id}/calibrate` | One or more images from a fixed camera → saved slot grid |
| GET | `/sites/{site_id}/slots` | Saved grid |
| PUT | `/sites/{site_id}/slots` | Manual correction of the grid |

Without `site_id` the grid is built from the same image the occupancy is computed on. With `site_id`
the saved grid is used; this is only correct for a camera that does not move.

What a request does, in order:

1. detects cars with the tiled model;
2. fits the local scale and cuts out the working zone;
3. builds the paint mask and finds the dividers;
4. assembles rows, continues them under cars and builds the slots;
5. matches the ground point of every detection to a slot;
6. returns the counts, the status and every slot polygon.

The response of `/occupancy/detect` for the first screenshot, with the slot list shortened:

```json
{
  "site_id": null,
  "total_slots": 6,
  "confirmed_slots": 6,
  "unconfirmed_slots": 0,
  "occupied": 3,
  "free": 3,
  "occupancy_rate": 0.5,
  "cars_in_zone": 7,
  "unassigned_detections": 4,
  "outside_zone_detections": 35,
  "calibration_status": "ok",
  "calibration_reason": null,
  "calibration_coverage": 1.0,
  "rows": [{"row": 0, "total": 6, "occupied": 3}],
  "slots": [{"slot_id": "r0_s00", "occupied": true, "confirmed": true, "conf": 0.8246, "source": "projected", "polygon": "..."}],
  "inference_ms": 838
}
```

Calibration status:

| Status | When | What the response contains |
|---|---|---|
| `ok` | at least 3 slots, and at least 60% of them confirmed | the full grid and the rate |
| `partial` | slots were found, but few of them or with low coverage | the grid, the rate and the reason in `calibration_reason` |
| `failed` | no scale (fewer than 5 cars), no markings or no rows | only the number of cars in the working zone, `occupancy_rate: null` |

`ok` describes the grid that was found, not the share of the car park it covers: three confirmed
slots out of forty real ones is still `ok`. How much of the car park the grid covers is what the
slot recall above measures.

### Tests

```bash
uv run pytest -q
```

The 21 tests are synthetic and need neither a GPU nor the model weights. Two manual checks need the
weights and the frames from the dataset:

```bash
uv run python tests/check_detector.py <dataset>/train/train
uv run python tests/check_markings.py <output folder>
```

## Project structure

```
parkingscan/
├── README.md
├── Build.md                         # the full specification the project is built by (in Russian)
├── LICENSE
├── pyproject.toml, uv.lock
├── configs/default.yaml             # every threshold of the service and of the grid algorithm
├── docker/
│   ├── Dockerfile                   # multi-stage CPU image
│   └── docker-compose.yml
├── models/
│   ├── cars_yolo26s_best.pt         # trained weights
│   └── config.json                  # inference contract: scale, tile, overlap, postprocessing
├── notebooks/CarsObjectDetection.ipynb   # how the model was trained
├── training/                        # training curves and validation mosaics
├── data/samples/                    # reference JSON files; the frames come from Kaggle
├── docs/screenshots/
├── src/parkingscan/
│   ├── cli.py                       # batch mode without the API
│   ├── config.py
│   ├── api/                         # FastAPI app, routes, schemas
│   └── core/
│       ├── detector.py              # YOLO + SAHI
│       ├── calibration.py           # slot grid from painted markings
│       ├── occupancy.py             # detections → slots
│       ├── geometry.py
│       └── render.py
└── tests/
```

## Honest limitations

- **On oblique handheld photos the grid is partial.** Markings are visible where slots are empty, and
  hidden by car bodies where slots are taken. This is what limits the slot recall to 0.50 and 0.43.
- **A row needs at least two visible dividers.** Without them it is not found, and its cars are
  reported as `unassigned_detections`.
- **Slots under cars are reconstructed, not seen.** They come from the step of the visible part of
  the row, not from their own markings.
- **Slot geometry can be poor on hard frames.** On one of the two reference frames the occupancy is
  assigned correctly, but several slots are skewed against the real lines or lie on open asphalt.
  Grid precision is checked by eye only; there is no metric for it.
- **`ok` is not coverage.** The status judges the grid that was found, not how much of the car park
  it covers.
- **A slot without painted markings is never found.**
- **Worn markings, snow, wet asphalt with glare and night images are not supported.**
- **The far field outside the working zone is not analysed.**
- **Detector false positives become occupied slots.** The detector occasionally takes a non-car
  object for a car; one such case, what appears to be a street lamp head, is known on the reference
  frames.
- **Tied to one kind of camera.** A wide-angle camera looking at the car park at an angle, as in the
  training data. The model is bound to that scale.
- **Tuned on four frames.** The thresholds of the grid algorithm were chosen on four sample frames
  and verified on two of them.
- **Saved grids assume a fixed camera.** The `site_id` mode has been checked for the mechanics only;
  there is no series of frames from one camera position in the data.
- **GPU in Docker and Linux hosts are not tested**, see the installation notes.
- **No database and no number plates yet.** Both are planned and absent from the code.

## Number plates need a separate camera

Plates cannot be read from an overview photo of a car park. This is geometry, not a model limitation:
reliable recognition needs about 100 px across the plate, and in an overview frame only one or two
cars right under the camera get that. The planned ANPR module therefore works from a separate
ground-level camera at the entrance, looking along the lane: plate width of at least 100 px (40 px is
the minimum), an angle to the plate of at most 30° horizontally and vertically, and a capture distance
of 5–15 m. This module is not implemented yet.

## Roadmap and ideas for improvement

- **Database of snapshots** — SQLite with slot states per snapshot and a batch endpoint with CSV export.
- **ANPR module** — tracking, best-shot selection, rectification, OCR and per-character voting over
  many frames of one vehicle, from a ground-level camera.
- **Seed a row from one divider plus the cars** — the cars that have no visible markings nearby are
  what keeps the recall at 0.50.
- **Measure grid precision** — hand-labelled slot polygons on a few frames instead of checking by eye.
- **Test the GPU image and a Linux host**, then publish a GPU build target.

## License

Released under **AGPL-3.0** — see [`LICENSE`](LICENSE). The detector is built on Ultralytics YOLO,
which is distributed under AGPL-3.0, so the whole project is released under the same license. Closed
commercial use requires a commercial Ultralytics license or a different detector.

The training data comes from the Kaggle competition
**[Cars object detection](https://www.kaggle.com/competitions/cars-object-detection)** and is subject
to its [Rules](https://www.kaggle.com/competitions/cars-object-detection/rules) — check them before
redistributing any frames.

---

<div align="center">
<sub>Built with Python · Ultralytics YOLO26 · SAHI · OpenCV · FastAPI — <b>ParkingScan — parking occupancy from painted markings</b><br>
Images are derived from the Kaggle competition "Cars object detection" · the rights to the original frames belong to its author.</sub>
</div>
