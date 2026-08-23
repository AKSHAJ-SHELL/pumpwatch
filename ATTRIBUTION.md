# Attribution and licensing

Three third-party works carry obligations that this project must discharge on
distribution. Each is recorded next to the code that uses it, so the obligation
travels with the code rather than living only in a document; this file collects
them in one place, which is what a data-availability statement and a code release
both need.

## TabPFN — Prior Labs License v1.1

> **Built with PriorLabs-TabPFN**

The Prior Labs License is Apache 2.0 with an added attribution clause (§10): a
product containing the weights must display the notice above. This project pins
`tabpfn` on the 2.x package line, which *is* the v2 model. Package version and
model version are not the same thing and the PyPI history invites the confusion —
the line runs 2.x then jumps to 6.x/7.x/8.x, where several model versions sit
behind a `ModelVersion` selector and v2 must be requested explicitly. There is no
3.x package.

The notice is asserted in code (`ATTRIBUTION_NOTICE` in
[tabpfn_clf.py](src/pumpwatch/gateway/tabpfn_clf.py)) and covered by a test, so
removing it breaks the suite rather than passing silently.

Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S.B.,
Schirrmeister, R.T., Hutter, F. (2025). Accurate predictions on small data with a
tabular foundation model. *Nature* 637, 319–326.

## ESPset — CC BY 4.0

Pellegrini, M., Varejão, F., Rodrigues, A., Mello, L.H.S. (2024). ESPset.
Mendeley Data, V3. DOI [10.17632/m268jsw339.3](https://doi.org/10.17632/m268jsw339.3)

CC BY 4.0 requires attribution and a statement of changes. The data are used
unmodified as downloaded; this project derives features from the published
order-normalised spectra (velocity in in/s, converted to mm/s at 25.4 mm/in) but
does not redistribute the dataset. Version 3 is the one used — **not** the `.1`
cited in some earlier work, which differs.

## Twente / 4TU centrifugal pump dataset — CC BY 4.0

Kumar, D. et al. (2023). Motor current and vibration monitoring dataset for
various faults in an E-motor-driven centrifugal pump. *Data in Brief* 51:109779.
DOI [10.4121/2b61183e-c14f-4131-829b-cc4822c369d0](https://doi.org/10.4121/2b61183e-c14f-4131-829b-cc4822c369d0)

Used unmodified as downloaded; a subset of the 20.8 GB archive is extracted
locally and features are derived from it. The dataset is not redistributed here.

## This project

The code in this repository is released under the MIT License; see
[LICENSE](LICENSE). No dataset files are included in the repository — every
loader raises with download instructions rather than inventing data, so the
licences above attach to what you download, not to what you clone.
