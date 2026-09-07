# Software and Data Availability

1. **Name of software:** `topology-aware-river-deeponet`
2. **Developer:** Ju Zhou
3. **Contact:** [404426902@qq.com](mailto:404426902@qq.com)
4. **Date first available:** September 7, 2026
5. **Software required:** Python 3.10 or later. Python dependencies are listed in [`requirements.txt`](requirements.txt). CUDA is optional and is used only for GPU acceleration.
6. **Program language:** Python
7. **Source code:** [https://github.com/zhou-j0313/topology-aware-river-deeponet](https://github.com/zhou-j0313/topology-aware-river-deeponet)
8. **Documentation:** Installation, testing, data preparation, and execution instructions are provided in the repository [`README.md`](README.md). Dataset schemas and units are documented in [`DATASET.md`](DATASET.md).
9. **Data required for local installation and use:** A limited, representative CSV subset is included in the `data/` directory so that users can inspect the required schemas and exercise the public baseline workflow. The complete research dataset is not included in the current preview release. See the Data Availability Statement below.

## Data Availability Statement

The source code and data currently deposited in this repository constitute a
streamlined preview release. They do not contain the complete research
implementation, the paper-specific composite loss, or the full training,
validation, and test datasets used in the associated study. The included code
and three representative events are provided to document the software
architecture, input formats, and baseline workflow. They are not intended to
reproduce every quantitative result reported in the manuscript.

The complete source code, complete research dataset, experiment configuration,
and additional reproducibility materials will be made openly available in this
repository after acceptance of the associated manuscript. Until that time,
materials beyond the current preview are not publicly available because the
study is under peer review.
