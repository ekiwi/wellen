# Wellen Wavform Library

[![Crates.io Version](https://img.shields.io/crates/v/wellen)](https://crates.io/crates/wellen)
[![docs.rs](https://img.shields.io/docsrs/wellen)](https://docs.rs/wellen)
[![GitHub License](https://img.shields.io/github/license/ekiwi/wellen)](LICENSE)
[![DOI](https://zenodo.org/badge/707242016.svg)](https://zenodo.org/doi/10.5281/zenodo.12774824)

`wellen` provides a common interface to read both FST and VCD waveform files.
The library is optimized for use-cases where only a subset of signals need to
be accessed, like in a waveform viewer.
VCD parsing uses multiple-threads.

## Overview

![Overview of wellen components](./wellen_overview.svg)
