# Load-test report

## Locust request summary

| Name | Requests | Failures | Median (ms) | 95th percentile (ms) | RPS |
|---|---:|---:|---:|---:|---:|
| /chat [multi_hop_composite] | 52 | 0 | 58 | 67 | 1.1761547370258296 |
| /chat [negative_out_of_scope] | 26 | 0 | 55 | 69 | 0.5880773685129148 |
| /chat [single_hop_factual] | 73 | 0 | 56 | 66 | 1.6511403039016455 |
| /chat [tool_required] | 46 | 0 | 58 | 66 | 1.0404445750613107 |
| /chat [unanswerable_from_corpus] | 25 | 0 | 55 | 66 | 0.565459008185495 |
| Aggregated | 222 | 0 | 57 | 67 | 5.021275992687196 |

## Per-node timing

| Node | Samples | Mean (ms) | p50 (ms) | p95 (ms) |
|---|---:|---:|---:|---:|
| execute | 255 | 8.8 | 6.0 | 17.0 |
| finalize | 259 | 1.1 | 1.0 | 2.0 |
| planner | 255 | 4.4 | 2.0 | 8.0 |
| synthesize | 255 | 2.2 | 2.0 | 5.0 |
| triage | 259 | 279.5 | 4.0 | 18.7 |
| verify | 255 | 3.0 | 2.0 | 6.3 |

## Telemetry

Metrics snapshots were not supplied.
