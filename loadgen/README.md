# Load Generator
This directory constains simple tools to generate FLOPS and memory load on the
given number of cores.

## FPGen 
```
cargo run --bin fpgen --release -- -h
```

The user can specify the input arguments for
- Number of cores to use with the `--cores` flag. The default is `1`.
- Single precision or double precision with the `--precision` flag. The default is `single` precision.
- The number of iterations to run with the `--iterations` flag. The default is `1000000000`.

### Example
```
cargo run --bin fpgen --release -- --cores 4 --precision double --iterations 1000000000
```

## MemGen
```
cargo run --bin memgen --release -- -h
```

The user can specify the input arguments for
- Number of cores to use with the `--cores` flag. The default is `3`.
- Number of cores to skip with the `--skip` flag. The default is `1`.

### Example
```
cargo run --bin memgen --release -- --cores 3 --skip 1
```

This will bind the memory generator to cores 2nd, 3rd and 4th core on socket 0.
