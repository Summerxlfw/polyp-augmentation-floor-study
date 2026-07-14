# Provenance

The three full training-pipeline files were pulled read-only from the completed
`xiat` formal-run workspace on 2026-07-14. Their server SHA-256 values were:

| File | Final server SHA-256 |
|---|---|
| `s1_loco_common.py` | `f8fbb8955115274744950ffec8112f43ce2c7aec9bcb7d999bd6d1b4092fd689` |
| `l1_official_arms_driver.py` | `4428be4432e270ed3054c28c6d788c02292fde3ffbc2a44c44bf3aa4a0749e40` |
| `run_smoke.py` | `a84e911780767dfa1535ac3ce2444d3aa9af9811f186c7603f4ee6184ad690a2` |

The public `src/training/s1_loco_common.py` differs from that snapshot only in
path configuration: user-specific server defaults were replaced by documented
environment variables. The formal augmentation functions, losses,
sampling ranges, model selection, batch size, and training budget were not
changed. `src/training/augmentations.py` is a standalone extraction of the four
spatial-formulation cells for direct testing.

The public CSVs omit the non-analytic `checkpoint` column because it contained
machine-specific paths. Numeric evidence and pairing identifiers are unchanged.
Both the source-raw and public-file checksums are recorded below.

| File | Source-raw SHA-256 | Public-file SHA-256 |
|---|---|---|
| `main_flat_cases.csv` | `783c98b0f3201706d2ddf2435f844977f32a92b922b534b92ea80b348e09e02a` | `c3cbc17622b207941f2deb86031d179b9663459d213eb29976649371e123bfc6` |
| `external_boundary_cases.csv` | `a8b24ca76968601a380278c000bd0a393f63160f11b2b61925baf50fce1fe7c8` | `7cd01aaac639edc6f7b44e8a6353d44a18108e850fd12f0de424e66893551e27` |
| `joint_affine_boundary_cases.csv` | `084fb26b7a8873747a3d58ad698124ef678d52b8ed3a63932064f549d8c7b7df` | `b852532d59490a4e3f6e393747de08664106e888c2f286e8c073e8916abcff09` |
| `factorial_4cell_flat_case.csv` | `d6c66b05e5c6884942578e97f2143b12bd7146061cabcf3a4de218d732057d46` | `35db1393865c7e15ea06c7a7bad7b4fd664cfb89569fe9b1b439756535c7362d` |
| `factorial_4cell_boundary_case.csv` | `79aee87a6e292d1b0b51cc881b50c7ad37c646d13dcf0770b37964e03d29448f` | `79aee87a6e292d1b0b51cc881b50c7ad37c646d13dcf0770b37964e03d29448f` |
| `full_zoo_eight_configuration_boundary_cases.csv` | `2b08f33296c9319648d5f48ae853295a87e00acf4974622e15ce45baba011961` | `2b25b37f3242a23b50b59e8db7aba482c0c79dfbdf1fc42102debd202e0fdf34` |

No raw endoscopy images, annotations, model weights, user credentials, private
hospital data, or server logs are included.
