# Metadata

This folder contains public dataset identifiers and labels copied from the experiment workspace. Machine-specific WSI and annotation paths were removed.

- `tcga_coad_read_95_slides.json`: 95-slide manifest consistent with the sample count stated in the current manuscript.
- `tcga_coad_read_canonical_manifest.json`: 117-slide manifest referenced by some canonical experiment materials.
- `tcga_muc_vs_nos_manifest.json`: auxiliary diagnosis-task manifest.
- `camelyon16_train_manifest.json`: official-training-partition manifest used by the project.
- `camelyon16_test_manifest.json`: held-out-test-partition manifest used by the project.

The 95-slide and 117-slide manifests correspond to different retained analyses described in the manuscript and are intentionally provided separately. Estimates produced from these manifests should not be treated as if they were derived from the same cohort.
