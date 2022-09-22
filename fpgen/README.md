# FLOPS Generator
This is a simple tool to generate FLOPS on the given number of cores. 

## Usage
```
cargo run --release 
```

The user can specify the input arguments for
- Number of cores to use with the `--cores` flag. The default is `1`.
- Single precision or double precision with the `--precision` flag. The default is `single` precision.
- The number of iterations to run with the `--iterations` flag. The default is `1000000000`.

## Example
```
cargo run --release -- --cores 4 --precision double --iterations 1000000000
```
