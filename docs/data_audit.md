# SDLCD Data Audit

## 1. Purpose

This document records the structure and properties of real lightcurve observations from SDLCD Data Release 3, to guide TumbleScope's ingestion, preprocessing and simulation modules.

## 2. Observation 1: COSPAR 00029A

**Filename:** `00029A_2_3_R_20190322_DATA.txt`

| Property | Value |
|---|---|
| COSPAR ID | 00029A |
| Observation date | 22 March 2019 |
| Observation start | 23:12:02 |
| Number of points | 273 (header-reported) |
| Reported period | 65.8 s |
| Reported period uncertainty | 0.005 s |
| Observation duration | 1459.3 s (24.32 minutes) |
| Exposure time | 2 s |
| Median sampling interval | 5.184 s |

### File format

The file contains comment lines beginning with `#`, followed by whitespace-separated numerical data.

Each numerical row contains six values:

1. Elapsed time in seconds

2. Exposure time in seconds

3. ADU (brightness measurement)

4. ADU uncertainty

5. Instrumental magnitude

6. Magnitude uncertainty

The header lists five column names, while the numerical rows contain six values. The second column is the exposure time in seconds. It matches the header's `Exp[s]` on every measurement row in all 47 files.

### Time convention

The first timestamp is 0 seconds, and the final timestamp is 1459.3 seconds.

The header duration is 24.3216 minutes, consistent with the numerical timestamps.

### Brightness convention

The file provides both ADU and instrumental magnitude measurements.

For magnitudes, smaller numerical values represent greater brightness. Preprocessing must account for this convention.

### Notes

- The original file is stored in `data/sdlcd_raw/`.

- The original data must be preserved.

## 3. Observation 2: COSPAR 00041B

**Filename:** `00041B_2_3_R_20191118_DATA.txt`

| Property | Value |
|---|---|
| COSPAR ID | 00041B |
| Observation date | 18 November 2019 |
| Observation start | 17:49:12 |
| Number of points | 248 (header-reported) |
| Reference period | Unknown (9999 special code) |
| Observation duration | 1095.55 s (18.26 minutes) |
| Exposure time | 1 s |
| Median sampling interval from the measurement rows | 4.060 s |

### File format

The numerical data contains six whitespace-separated columns, consistent with observation 00029A.

The columns represent elapsed time, exposure time, ADU, ADU uncertainty, instrumental magnitude, and magnitude uncertainty.

### Reference period handling

The header contains `Per[s] = 9999`, which is a reserved internal code rather than a measured rotation period.

This observation must not be assigned a reference period of 9999 seconds.

### Notes

- The original file is stored in `data/sdlcd_raw/`.
- The first timestamp is 0 seconds.
- The last timestamp is 1095.55 seconds.
- The header duration of 18.2592 minutes agrees with the timestamps.
- The numerical column layout matches the first observation. The header's `MedPD[s]` value is 3.8016 s. The median of the positive intervals between the recorded timestamps is 4.060 s.


## 4. Dataset Statistics

The SDLCD audit was conducted using 47 lightcurve files representing multiple space-debris objects.

### Dataset composition

- Total lightcurves: 47
- Lightcurves with valid reported periods: 29
- Distinct objects with valid reported periods: 25
- Lightcurves with excluded period codes: 18

### Observation duration

| Statistic | Duration (seconds) |
|---|---:|
| Minimum | 220.838 |
| Maximum | 13078.2 |
| Median | 1199.23 |
| Mean | 1484.411 |

### Sampling interval

| Statistic | Median sampling interval (seconds) |
|---|---:|
| Minimum | 3.024 |
| Maximum | 17.799 |
| Median | 4.0608 |
| Mean | 4.875 |

### Reported period statistics

| Statistic | Reported period (seconds) |
|---|---:|
| Minimum | 2.912 |
| Maximum | 445.881 |
| Median | 31.472 |
| Mean | 77.466 |

The period statistics were calculated using the 29 observations with valid reported periods.

### Special period codes

The following period codes were identified in the original SDLCD file headers:

| Code | Number of files |
|---|---:|
| 9999 | 10 |
| 6000 | 4 |
| 5000 | 3 |
| 7000 | 1 |

SDLCD identifies these values as reserved internal type codes. The audit excludes them from the reference-period column. [SDLCD Data Release 3 field definitions](https://www.sdlcd.space-debris.sk/)

The [per-file audit table](sdlcd_audit.csv) records filename, object ID, row and header point counts, duration, median cadence, and the apparent reference period where available. Reference periods are for evaluation only.

### Header and time convention

The eight comment records contain the observation identifier, observation start time, duration, a separator, statistics headings, statistics values, another separator, and measurement headings. Read `Per[s]` and `Points` from the statistics values under their corresponding headings. Each record has a distinct role.

The measurement timestamps already represent seconds elapsed from the first sample. The header duration is in minutes. Multiplying it by 60 agrees with the measured span in every file within 0.006 s.

### Interpretation

The collected observations have varying durations and sampling intervals. Several files contain irregular sampling gaps, which may affect period estimation.

The earlier proposed **6–500 s** search range includes 27 of 29 reference periods. It excludes 2.912 s and 5.868 s, both of which are sampled sparsely relative to their period. Three in-range references (7.1768, 7.2218, and 10.7043 s) also have fewer than two samples per cycle at their median cadence. The CP1 decision in Section 6 supersedes this proposal.

At 6 s, none of the 47 files reaches two samples per cycle at its median cadence. At 500 s, only 14 files have a baseline long enough to span three cycles. Even the 445.881 s reference spans only 2.61 cycles, including time across a gap. These are exploratory sampling checks. Irregular timestamps, gaps, exposure time, and competing periods can all affect recovery. Recovery performance still requires injections into real sampling patterns; setting search bounds does not establish recoverability.

## 5. Data Quality and Sampling Gaps

All **47 files** decoded successfully. All **10,719 measurement rows** had six finite numeric values, and every file matched its header point count. No malformed rows, missing brightness values, or decreasing timestamps were found.

- One complete measurement row repeats at **368.064 s** in `15010C_1_2_3_R_20190531_DATA.txt`. Both rows count toward the file's header total.
- The ADU column contains **2,526 negative measurements across 18 files**. These are finite values. The instrumental magnitude column is finite throughout.
- There are **73 large gaps across 41 observations**. A gap counts when the time between consecutive samples exceeds three times that observation's median positive sampling interval. This threshold is for the audit.

| File | Large gaps |
|---|---:|
| `12003B_1_2_3_4_5_R_20190910_DATA.txt` | 6 |
| `12003B_4_6_R_20190809_DATA.txt` | 5 |
| `15010C_1_2_3_R_20190531_DATA.txt` | 5 |
| `01038B_2_3_R_20190322_DATA.txt` | 4 |

The largest gap is **6,344.79 s** in the 2019-09-10 file for `12003B`.

## 6. CP1 recovery handoff (22 November 2026)

The 47 originals were obtained from the official SDLCD DR3 site's per-observation
`static/data/` links discovered in its public application and catalogue. No bulk
endpoint was assumed. `data/sdlcd_raw/download_manifest.json` preserves each
source URL, catalogue record, object identity, reported period field, byte count,
SHA-256 digest and retrieval time. Raw observations remain Git-ignored.
The site requests citation and acknowledgment for academic use; acknowledge
SDLCD and cite its [linked publication](https://www.sciencedirect.com/science/article/pii/S0273117720300727).
The user authorized local research acquisition under that notice. Redistribution
terms remain unconfirmed, so the planned 3–5 redistributable sample files have
not been added to `data/samples/`.

The verified extrema of all 29 valid reported apparent periods define CP1 search
coverage: **2.912–445.881 seconds**, inclusive. This avoids dropping either short
reference and adds no unsupported range beyond this pilot. These are population
coverage bounds, not a claim that every observation resolves every in-range
period. `configs/period_bounds.yaml` is the sole editable source, following
`CONTRACTS.md` Section 10. Reserved codes are excluded before this calculation.

The evaluation-only reference table contains **29 observations from 25 objects**:
18 validation and 11 test observations. SHA-256 of seed 0 plus object ID assigns
whole objects deterministically. These labels are not attached to observations,
sampling profiles or inference inputs. All 47 observations yield sampling
profiles under the schema in `CONTRACTS.md` Section 9; no brightness arrays or
reference periods are included.

Stage 1 thresholds in `configs/preprocess.yaml` are CP1 operating choices,
checked against clean, noisy, short and gapped controls: minimum 21 valid points,
maximum gap/baseline 0.25, maximum missing fraction 0.2 and minimum robust
amplitude/noise proxy 3. Duplicate samples and outliers are retained and flagged;
gaps exceed three positive median intervals, and bright outliers exceed six MAD.
The difference-based noise proxy can include signal variation; these thresholds
are not a measured real-data period-recovery calibration.

The clean synthetic pilot uses the observed minimum and 90th-percentile duration
(220.838 and 1851.174 seconds). Adequately covered sources are conditioned on at
least three cycles of the largest configured period; under-covered sources use
the unconditioned duration range and must fail the configured cycle count for
their clean parent repeat. Cadence is one eighth of the shortest configured
period (0.364 seconds), deliberately denser than the real observations. Mixture
parents use the upper half of the configured cycle range to keep dense repeat
validation tractable. These are explicit pilot design choices, not claims that
the raw observations have these sampling properties. The generator records the
full configuration, quotas, rejection counts, seed, source hashes and label
policy. Real-profile degradation and the shared feature pipeline remain CP2.

Local acceptance evidence is recorded in `docs/results/runtime_cp1_recovery.json`.
It is checked by `python -m scripts.validate_cp1 --pilot data/synthetic/cp1_pilot`,
which reads existing artifacts without downloading or regenerating them.
