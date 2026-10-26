Verbatim copies of three NVIDIA flash_chord files (Apache-2.0, SPDX headers intact) from
`nvidia-isaac/video_to_data` at `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`, under
`robotic_grounding/flash_chord/src/flash_chord/`:

- `objectives/recon_body.py`                        -> `objectives_recon_body.py`
- `configs/objective/recon_body.yaml`               -> `configs_objective_recon_body.yaml`
- `configs/curriculum/recon_body_flash_sac.yaml`    -> `configs_curriculum_recon_body_flash_sac.yaml`

They are here so `tests/test_object_orientation.py` can prove `brev/patch_object_orientation.py`
applies cleanly to the real upstream text on the Jetson, before any paid box runs it. flash_chord
is not installed locally -- it ships inside the training image -- so without these the patch could
only be tested by spending money.
