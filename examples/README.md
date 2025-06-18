# `feadme` Example

This directory contains example configurations for using the `feadme` package, 
which is designed to facilitate modeling of disk emission in spectral data. 

In this directory, you will find an example JSON configurations that illustrates
how to define the disk and line profiles, as well as how to set up shared 
parameters between different profiles.

Also included is a data set for the AGN double-peaked emitter ZTF18aahiqst.
To run the example, you can use the following command:

```bash
feadme template.json data.csv --output-path=./output --num-warmup=1000 --num-samples=1000 --num-chains=2 --pre-fit
```