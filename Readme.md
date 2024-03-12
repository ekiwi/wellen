# Wellen Wavform Library

[![Crates.io Version](https://img.shields.io/crates/v/wellen)](https://crates.io/crates/wellen)
[![docs.rs](https://img.shields.io/docsrs/wellen)](https://docs.rs/wellen)


`wellen` provides a common interface to read both FST and VCD waveform files.
The library is optimized for use-cases where only a subset of signals need to
be accessed, like in a waveform viewer.
VCD parsing uses multiple-threads.

## Overview

![Overview of wellen components](./wellen_overview.svg)
