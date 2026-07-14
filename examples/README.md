# `feadme` Example

This directory contains example configurations for using the `feadme` package, 
which is designed to facilitate modeling of disk emission in spectral data. 

In this directory, you will find an example JSON configuration that illustrates
how to define the disk and line profiles, as well as how to set up shared 
parameters between different profiles.

Also included is a data set for the AGN double-peaked emitter ZTF18aahiqst.
To run the example, you can use the following command:

```bash
feadme run \
  --template-path template.json \
  --data-path data.csv \
  --output-path ./output \
  --num-warmup 1000 \
  --num-samples 1000 \
  --num-chains 2 \
  --init-method jax-lsq \
  --integrator mixed
```

If you are using a CPU and want JAX to treat it as several distinct devices,
set `FORCE_DEVICE_COUNT` to the number of devices
you want to simulate, e.g.

```bash
export FORCE_DEVICE_COUNT=2
feadme run \
  --template-path template.json \
  --data-path data.csv \
  --output-path ./output \
  --num-warmup 1000 \
  --num-samples 1000 \
  --num-chains 2 \
  --init-method jax-lsq \
  --integrator mixed
```
