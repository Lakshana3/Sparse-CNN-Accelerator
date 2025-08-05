# ASIC Implementation of a Sparse CNN Accelerator Block using OpenLane and Sky130 PDK

## Overview
This repository contains the RTL design, simulation, and ASIC implementation flow for a sparsity-aware, pipelined Radix-4 Booth multiplier block, optimized for efficient convolutional neural network (CNN) acceleration. The project demonstrates how dynamic sparsity gating, Booth pre-encoding, and pipelined MAC units can reduce power and area for next-generation AI hardware.

## Features
- Sparsity-aware multiplier: Skips zero-valued operations using mask-based gating.
- Radix-4 Booth pre-encoding: Reduces the number of partial products for energy and area savings.
- Pipelined MAC array: Supports high throughput and efficient clock gating.
- Verification: Functional testing with Icarus Verilog; simulation waveforms viewable in EPWave.
- Open-source ASIC flow: RTL-to-GDSII flow managed with OpenLane2 and SkyWater 130nm PDK.

## Directory Structure

├── Code/                # Verilog RTL source files and testbenches

├── openlane/           # OpenLane2 configuration and results

├── docs/               # Project report and documentation

└── README.md           # This file

## Results
- Area: 75,321.5 μm² (optimized block)
- Utilization: 53.9%
- Total Power: 11.44 mW
- Timing: 0 setup violations, WHS = +0.23 ns
- See report for more results and analysis.

## References
Key papers and documentation used in this project are listed in the /docs folder and at the end of the report.

## License
This project is released under the MIT License.

## Acknowledgements
- Faculty and guides from APEC
- OpenLane, SkyWater, and open-source EDA communities
