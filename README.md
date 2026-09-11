# fridge-watcher

Turns Frigate camera events at the fridge into inventory updates in Supabase,
for [FridgeFriend](#).

A Reolink camera on the ceiling watches the fridge door and the counter in
front of it. Frigate records a clip whenever something happens there. This
service picks up the clip, samples N frames from it, asks Claude what item
moved and **which way it was going**, and writes the result to Supabase —
either straight into inventory, or into a review queue when it is not sure.

```
Frigate ──MQTT──▶ mqtt_listener ──▶ frigate (clip download)
                                        │
                                        ▼
                                  frame_extractor  ──▶ ./captures/{event_id}/
                                        │
                                        ▼
                                     vision  (Claude, N frames in one call)
                                        │
                                        ▼
                                      store  ──▶ Supabase
```

---

## How direction is decided

The hard part is knowing whether an item came **out** of the fridge or went
**in**. This service does *not* diff a "before" and "after" picture of the
counter — items never get put down in the same place twice, so that comparison
is worthless.

Instead all N frames go to the model in a single call, in order, each labelled
`Frame 1 of 6` … `Frame 6 of 6`. The model locates the fridge door (it is the
one thing that does not move), then tracks the item's position relative to the
door across the sequence:

| Trajectory | Verdict |
| --- | --- |
| Starts near the door, gets farther away | `OUT` |
| Starts away from the door, gets closer | `IN` |
| Only two or three sightings, but consistent | Still called, with lower confidence |
| No identifiable item in any frame | `NO_ITEM`, nothing is written |

The item does not have to be visible in every frame — occlusion by a body or
the open door is expected.

---

## 1. Frigate configuration (your side)

Everything runs on one always-on machine — a Mac Mini, a mini PC, a Pi. The
repo root has a `docker-compose.yml` that brings up all three pieces (Frigate,
a Mosquitto broker, and this service) and `frigate/config.yml` for the camera.

```bash
git clone https://github.com/daltarescu-4581/ZoraNeon.git
cd ZoraNeon
cp .env.example .env          # camera password, API keys
$EDITOR frigate/config.yml    # camera IP + Reolink username, in both paths
docker compose up -d
docker compose logs -f
```

Open `http://<that machine>:8971` and confirm you can see the camera.

Inside the compose network the three containers find each other by name, so
`FRIGATE_URL` and `MQTT_HOST` are set for you and the values in `.env` are
only used when running the CLI outside Docker.

### Running it on a Mac (Mac Mini, iMac, anything always-on)

Works well, with three differences from Linux:

**No hardware video decoding.** Docker on macOS runs containers inside a Linux
VM that cannot see VideoToolbox, so ffmpeg decodes on the CPU. For one camera
this barely matters: recording the main stream is a stream copy (no decoding
at all), and only the small detect stream is actually decoded. Leave the
`devices:` block in `docker-compose.yml` commented out — `/dev/dri` does not
exist on a Mac and will stop the container from starting.

**No Coral TPU.** USB passthrough into the Docker VM is not supported on
macOS. Not a problem here: one camera at 5fps on the CPU detector is fine, and
`TRIGGER_MODE=motion` skips the detector entirely.

**The Mac will fall asleep and stop watching your fridge.** This is the one
that actually bites. Turn sleep off:

```bash
sudo pmset -a sleep 0 displaysleep 0 disksleep 0
sudo pmset -a autorestart 1      # come back after a power cut
sudo pmset -a womp 1             # wake for network access
pmset -g                         # confirm: sleep should read 0
```

Then make sure Docker itself comes back after a reboot. Docker Desktop needs a
logged-in desktop session, which means enabling automatic login
(System Settings → Users & Groups → Automatically log in as). If you would
rather not do that, [Colima](https://github.com/abiosoft/colima) is a headless
Docker runtime that starts without a GUI session:

```bash
brew install colima docker docker-compose
colima start --cpu 2 --memory 4 --disk 60
brew services start colima        # survives reboot, no auto-login needed
```

One networking note: the camera streams continuously, so put the Mac on
**wired ethernet** if you can. Continuous RTSP over marginal Wi-Fi shows up as
corrupted clips and confusing model output rather than as an obvious failure.

### Getting the Reolink stream

Reolink exposes two streams, and Frigate wants both:

```
rtsp://USER:PASSWORD@<camera-ip>:554/h264Preview_01_main   # recorded -> our frames
rtsp://USER:PASSWORD@<camera-ip>:554/h264Preview_01_sub    # motion detection only
```

Use `h265Preview_01_main` if the camera records in H.265; the sub stream
usually stays H.264 either way.

Before configuring anything, prove the stream plays — `ffplay "rtsp://…"` or
VLC's Open Network Stream. Nothing downstream can work until it does.

Four things that reliably go wrong:

- **RTSP is disabled by default** on recent Reolink firmware. Turn it on under
  Settings → Network → Advanced → Port Settings.
- **Give the camera a DHCP reservation.** If its IP changes, Frigate stops
  seeing it and says nothing useful about why.
- **Use a letters-and-numbers password.** Special characters have to be
  URL-encoded inside an RTSP address and it is miserable to debug.
- **Make a separate Reolink user** for Frigate rather than reusing admin.

Detection quality note: the sub stream only answers "did something move?".
The frames Claude actually sees are cut from the recorded main stream, so a
grainy sub stream does not hurt item identification.

### Drawing the zone

The one thing that cannot be copied from a file. In the Frigate UI:
**Settings → Mask & Zone Editor → Zones**, draw a polygon named
`fridge_door`, and paste the coordinates it produces into `config.yml`.

Draw it over the door opening **plus roughly a hand's width of counter in
front of it**. Too tight and an item's "near the door" frames get clipped
off, which flattens the trajectory and leaves the model nothing to measure
direction against.

### Two trigger modes

`TRIGGER_MODE` switches fridge-watcher between two ways of noticing that
something happened, with no code change.

**`events` (default)** — subscribes to `frigate/events` and acts on
`type == "end"` when `after.entered_zones` contains your `ZONE_NAME`. Clean,
precise, and gives you Frigate's own clip.

**`motion`** — subscribes to `frigate/{camera}/motion` and treats each `ON` →
`OFF` span as one trigger, pulling the recording for exactly that window from
`/api/{camera}/start/{start}/end/{end}/clip.mp4`.

Why motion mode exists: **Frigate's detector tracks `person`, not groceries.**
An arm reaching into the fridge frequently does not register as a person, so
the event never fires and the item is missed entirely. Motion mode fires on
anything that moves. It is noisier — expect more `NO_ITEM` results — but it
does not miss. Start on `events`, and if you find yourself opening the fridge
and seeing nothing in the log, switch to `motion`.

Motion windows shorter than 1s are ignored and windows longer than 45s are
clamped to their last 45s (constants in `config.py` — they describe fridge
physics rather than a deployment choice).

Two things to check before switching to motion mode:

- **Draw a motion mask** over anything that moves but is not somebody at the
  fridge — a window, a doorway, a TV. In events mode a stray trigger is
  filtered out by the zone; in motion mode it becomes a Claude API call.
- **Recordings must still cover the window.** `frigate/config.yml` uses
  `record.retain.mode: motion`, which keeps exactly the segments motion mode
  asks for. If clip downloads start failing on windows you know happened,
  switch that to `all` and give it more disk.

---

## 2. Database

Run the migration against your Supabase project:

```bash
psql "$SUPABASE_DB_URL" -f migrations/0001_inventory.sql
# or paste it into the Supabase SQL editor
```

It creates `inventory_items`, `inventory_events`, and an
`apply_inventory_event()` function.

The function matters: it logs the event row and moves the stock **in one
transaction**, with `INSERT … ON CONFLICT (event_id) DO NOTHING` as the
idempotency guard. A redelivered MQTT message therefore cannot double-count,
and a crash cannot log an event without moving stock or vice versa.

If you do not run the migration, the service falls back to two client-side
round trips, logs a warning, and still refuses to double-count — but the write
is no longer atomic. Run the migration.

Write rules:

| Model output | `inventory_events` | `inventory_items` |
| --- | --- | --- |
| confidence ≥ `CONFIDENCE_THRESHOLD` | row, `status = applied` | quantity delta applied |
| confidence < threshold | row, `status = pending_review` | untouched |
| unparsable response | row, `status = pending_review` | untouched |
| `NO_ITEM` | nothing | untouched |

`OUT` subtracts, `IN` adds. Stock is clamped at zero: taking out something we
never saw go in is a gap in our history, not a negative fridge.

---

## 3. Running it

Under docker compose (section 1) the service is already running — that is the
normal deployment. Useful commands:

```bash
docker compose logs -f fridge-watcher
docker compose restart fridge-watcher      # after editing .env
docker compose up -d --build               # after changing the code
```

Logs are JSON lines on stdout, one object per line, so they pipe into `jq`:

```bash
docker compose logs -f fridge-watcher | jq -c 'select(.msg=="inventory_write")'
```

### Outside Docker

You need this on a laptop to use `replay` and `replay-all`, and it is also a
perfectly good way to run the service itself:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
$EDITOR .env          # here FRIGATE_URL and MQTT_HOST do matter -- point
                      # them at the machine running Frigate

python -m fridge_watcher run
```

On macOS the command is `python3` until the venv is activated; afterwards
plain `python` works.

### systemd

For a Linux host running the service natively rather than in a container:

```ini
[Unit]
Description=fridge-watcher
After=network-online.target

[Service]
WorkingDirectory=/opt/fridge-watcher
EnvironmentFile=/opt/fridge-watcher/.env
ExecStart=/opt/fridge-watcher/.venv/bin/python -m fridge_watcher run
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
journalctl -u fridge-watcher -o cat | jq .
```

---

## 4. Replay mode

You should never have to stand at the fridge to test a change.

**Every live event saves its frames** to `./captures/{event_id}/` as
`frame_01.jpg … frame_NN.jpg`, plus a `result.json` with the model's verdict.
That directory *is* your regression set.

### Replay a single clip

```bash
# Dry run — prints the verdict, writes nothing to Supabase
python -m fridge_watcher replay ~/clips/oatmilk-out.mp4

# Persist it
python -m fridge_watcher replay ~/clips/oatmilk-out.mp4 --write

# Try a different frame count without touching .env
python -m fridge_watcher replay ~/clips/oatmilk-out.mp4 --frames 10
```

```
capture      oatmilk-out
item         oat milk carton  (dairy_alternative)
quantity     1
direction    OUT
confidence   0.82
reasoning    carton is at the door edge in Frame 2 and near the sink by Frame 5
status       applied
frames       captures/oatmilk-out
supabase     dry run (pass --write to persist)
```

### Score the whole regression set

Hand-label your captures once in an expectations file (see
`expectations.example.json`):

```json
{
  "1727213456.123456-abc12d": {
    "item": "oat milk carton",
    "direction": "OUT",
    "quantity": 1,
    "aliases": ["oat milk", "carton of oat milk"]
  }
}
```

`aliases` exist because "oat milk" and "carton of oat milk" are both correct
for inventory purposes; name matching is deliberately loose (containment or a
majority of shared tokens).

Then:

```bash
python -m fridge_watcher replay-all ./captures --expect expectations.json
```

```
capture                            expected                   got                         conf  result
------------------------------------------------------------------------------------------------------
1727214012.987654-ef34gh           IN leftovers container     OUT leftovers container     0.81  WRONG: direction  <-- would have been applied
1727213456.123456-abc12d           OUT oat milk carton        OUT oat milk carton         0.88  OK

captures scored      2
direction accuracy   50.0%
item accuracy        100.0%
overall accuracy     50.0%
mean confidence      0.85
false applies        1   (wrong AND above threshold -- these corrupt inventory silently)
missed applies       0   (right but below threshold -- review-queue noise)
```

The two numbers to watch when tuning are the last two. **False applies** are
the expensive failure — wrong *and* confident, so they land in your inventory
with nobody looking. Missed applies are only annoying.

The vision call runs at `temperature=0`, so an unchanged prompt on unchanged
frames gives you the same answer — which is what makes this a usable
before/after measurement.

Useful flags:

```bash
--json                    # machine-readable report
--only <capture_id>       # score one capture (repeatable)
--fail-under 0.9          # exit non-zero below this accuracy, to gate a change
```

---

## 5. Tests

```bash
pip install pytest
python -m pytest
```

The suite covers the parts that are expensive to get wrong and awkward to test
by hand: defensive parsing of model output, frame sampling (including a
regression guard that sampled frames are genuinely different moments), event
and motion filtering, threshold routing, and idempotency on both the RPC and
the fallback write path. The Anthropic and Supabase clients are stubbed —
nothing in the suite touches the network.

---

## 6. Configuration reference

| Variable | Default | Notes |
| --- | --- | --- |
| `FRIGATE_URL` | `http://localhost:5000` | No trailing slash. |
| `FRIGATE_CAMERA_NAME` | `fridge` | Must match Frigate's `config.yml`. |
| `MQTT_HOST` / `MQTT_PORT` | `localhost` / `1883` | |
| `MQTT_USER` / `MQTT_PASSWORD` | unset | Omit for an anonymous broker. |
| `TRIGGER_MODE` | `events` | `events` or `motion`. |
| `ZONE_NAME` | `fridge_door` | Only events entering this zone are processed. |
| `FRAME_COUNT` | `6` | Frames sent per clip. Minimum 2. |
| `CONFIDENCE_THRESHOLD` | `0.75` | At or above → applied; below → review queue. |
| `ANTHROPIC_API_KEY` | — | Required. |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Optional override. |
| `SUPABASE_URL` | — | Required (except `replay` without `--write`). |
| `SUPABASE_SERVICE_KEY` | — | Service role key; writes bypass RLS. |
| `CAPTURES_DIR` | `./captures` | Frames + `result.json` per event. |

`--trigger-mode` on `run` overrides `TRIGGER_MODE` for one invocation, which is
handy when you are deciding which mode your camera needs.

---

## 7. Layout

| File | Responsibility |
| --- | --- |
| `config.py` | Environment → immutable `Config`. The only module that reads `os.environ`. |
| `mqtt_listener.py` | Frigate MQTT subscription, both trigger modes, worker queue. |
| `frigate.py` | Clip and recording downloads, with backoff. |
| `frame_extractor.py` | N evenly spaced JPEG frames from a clip; save/load capture dirs. |
| `vision.py` | The prompt, the frame message, defensive JSON parsing. |
| `store.py` | Supabase writes, idempotency, threshold routing. |
| `pipeline.py` | Ties it together; shared by live, `replay` and `replay-all`. |
| `evaluate.py` | Scoring for `replay-all`. |
| `cli.py` | Argument parsing and output. |

Deployment lives in `docker-compose.yml` (all three services), `Dockerfile`
(this service) and `frigate/config.yml` (the camera).

Frame extraction uses OpenCV (`opencv-python-headless`) rather than ffmpeg: one pip install with no system
binary to provision, frames come back as arrays we JPEG-encode in process, and
"evenly spaced frames" is a counting problem in OpenCV versus timestamp
arithmetic in ffmpeg. See the module docstring for the full reasoning.

---

## 8. Troubleshooting

**Nothing happens when I open the fridge.** Re-run with
`--log-level DEBUG` and look for `event_outside_zone`: that means the event
fired but your polygon did not cover where the action was, so redraw the zone
wider. If there is no line at all even at DEBUG, Frigate never detected a
person — switch to `TRIGGER_MODE=motion`.

**Frigate itself sees nothing.** Check the camera before the code: does the
Debug view in the Frigate UI show motion boxes when you open the fridge? If
not, the problem is the stream or the motion threshold, and nothing in
fridge-watcher will help.

**`clip_retry` then `ClipUnavailable`.** Frigate had not finished writing the
mp4. The service retries 5 times with exponential backoff; if it still fails,
confirm `record` is enabled for the camera and that clips exist in the Frigate
UI.

**Everything comes back `NO_ITEM`.** Usually the camera angle: if the fridge
door is out of frame the model has no anchor to measure the trajectory
against. Pull a capture directory up and look at the frames — that is what the
model saw.

**Everything lands in `pending_review`.** Confidence is systematically low.
Look at `reasoning` in `inventory_events`; it names the frames the model used,
which usually points at the fix (more frames, wider zone, better angle).
