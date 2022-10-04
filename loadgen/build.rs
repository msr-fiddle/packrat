fn main() {
    cc::Build::new().file("avx512f.S").compile("flopsgen");
}
