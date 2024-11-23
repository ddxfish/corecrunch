# CoreCrunch

A comprehensive CPU benchmark tool written in Rust that tests both low-level CPU operations and real-world computational tasks. The benchmark runs tests across different core counts to measure parallel processing efficiency and scaling, providing detailed metrics for each test including execution time, performance metrics (GFLOPS, GB/s), and parallel efficiency. Finally a Rust app with zero compiler warnings. This one came out great. 

## Tests

### Low Level Tests
* FFT (Fast Fourier Transform)
* SHA-256 Hashing
* Matrix Multiplication
* Floating Point Operations

### Real World Tests
* Monte Carlo Pi Calculation
* Prime Number Sieve
* N-Body Simulation

## Installation

Download the latest release from the [Releases](https://github.com/ddxfish/corecrunch/releases) page.

Or build from source:
```bash
git clone https://github.com/ddxfish/corecrunch
cd corecrunch
cargo build --release
```

## Usage

```
Usage: corecrunch.exe [OPTIONS]

Options:
  -p, --processes <PROCESSES>  Number of processes to use (default: number of CPU cores)
  -i, --intensity <INTENSITY>  Intensity level (1-10, default: 5) [default: 5]
      --low-level-only        Run only low-level benchmarks
      --real-world-only       Run only real-world benchmarks
  -h, --help                  Print help
  -V, --version              Print version
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.