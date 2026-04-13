# OGRD

Implementation of OGRD for open-world graph classification.

## Code Organization

The code is split into four scripts by dataset type:

1. **Binary molecular datasets** — `BZR-COX2`, `PTC_MR-MUTAG`, `AIDS-DHFR`
2. **Multi-class TU datasets** — `MSRC_9`, `Synthie`, `COLLAB`
3. **Temporal graph dataset** — `FB-TR`
4. **Fake news dataset** — `FakeNews`

Pick the script matching your target dataset.

All hyperparameters are set inside the script. Edit the script directly to change them.
