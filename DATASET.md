# Dataset

The repository includes three compact benchmark events for demonstrating the
data pipeline and training workflow. All files use UTF-8 encoded CSV.

## Network-level files

### `data/cross_sections.csv`

| Column | Unit | Description |
|---|---|---|
| `section_id` | - | Integer cross-section identifier |
| `station_x` | m | Lateral station coordinate |
| `elevation_z` | m | Bed or bank elevation |

Each section is represented by multiple station-elevation points.

### `data/river_network.csv`

| Column | Unit | Description |
|---|---|---|
| `reach_name` | - | Reach identifier |
| `section_id` | - | Cross-section identifier |
| `distance_to_next_m` | m | Distance to the next row in the same reach |
| `manning_n` | - | Manning roughness coefficient |

Rows are ordered from the head to the tail of each reach. A physical section
may occur in multiple reaches when it represents a junction endpoint.

### `data/boundary_sections.csv`

| Column | Unit | Description |
|---|---|---|
| `upper_boundary_section` | - | Section with prescribed discharge |
| `lower_boundary_section` | - | Section with prescribed water level |

## Event files

Each `data/case_XXX/` directory contains the following files.

### `boundary_conditions.csv`

| Column | Unit | Description |
|---|---|---|
| `section_id` | - | Boundary-section identifier |
| `boundary_type` | - | `upstream` or `downstream` |
| `time_hours` | h | Time relative to event start |
| `discharge_m3s` | m3/s | Boundary discharge |
| `water_level_m` | m | Boundary water level |

The model conditions on upstream discharge and downstream water level.

### `initial_conditions.csv`

| Column | Unit | Description |
|---|---|---|
| `section_id` | - | Cross-section identifier |
| `initial_discharge_m3s` | m3/s | Initial discharge |
| `initial_water_level_m` | m | Initial water level |

### `lateral_inflow.csv`

| Column | Unit | Description |
|---|---|---|
| `section_id` | - | Receiving cross-section identifier |
| `time_hours` | h | Time relative to event start |
| `discharge_m3s` | m3/s | Lateral inflow |

The bundled examples contain no lateral inflow, so these files contain only the
header. Add rows when lateral sources are available.

### `targets.csv`

| Column | Unit | Description |
|---|---|---|
| `section_id` | - | Cross-section identifier |
| `time_hours` | h | Time relative to event start |
| `water_level_m` | m | Reference water level |
| `discharge_m3s` | m3/s | Reference discharge |

Targets may be used for full or sparse supervision. For validation events, the
default script keeps them separate from model inputs and uses them only to
calculate evaluation metrics.

### `section_mapping.csv`

This provenance table maps model sections to source reach names and ordering.
It is included for traceability and is not read by the training script.

## Data split

- Training: `case_001`, `case_002`
- Validation: `case_003`

No claim is made that three events are sufficient for production deployment.
They are provided to exercise the complete workflow and document the expected
schema. Larger studies should define independent training, validation, and test
sets and report the split explicitly.
