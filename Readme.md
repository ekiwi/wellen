# Wellen Wavform Library

`wellen` provides a common interface to read both FST and VCD waveform files.
The library is optimized for use-cases where only a subset of signals need to
be accessed, like in a waveform viewer.
VCD parsing uses multiple-threads.

## Overview

![Overview of wellen components](./wellen_overview.svg)
