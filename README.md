# `hyperopt`

Tree-of-Parzen-estimators hyperparameter optimization for Rust

[![Documentation](https://img.shields.io/docsrs/hyperopt?style=for-the-badge)
](https://docs.rs/hyperopt)
[![Check status](https://img.shields.io/github/actions/workflow/status/eigenein/rust-hyperopt/check.yaml?style=for-the-badge)]((https://github.com/eigenein/rust-hyperopt/actions/workflows/check.yaml))
[![Code coverage](https://img.shields.io/codecov/c/github/eigenein/rust-hyperopt?style=for-the-badge)
](https://app.codecov.io/gh/eigenein/rust-hyperopt)
![Maintenance](https://img.shields.io/maintenance/yes/2024?style=for-the-badge)

## Example

```rust
use std::f64::consts::{FRAC_PI_2, PI};

use approx::assert_abs_diff_eq;
use fastrand::Rng;
use ordered_float::OrderedFloat;

use hyperopt::Optimizer;
use hyperopt::kde::Component;
use hyperopt::kernel::{Epanechnikov, Uniform};

fn main() {
    let min = OrderedFloat(FRAC_PI_2);
    let max = OrderedFloat(PI + FRAC_PI_2);
    let mut optimizer = Optimizer::new(
        min, max,                                               // parameter search range
        Component::<Uniform, OrderedFloat<f64>>::new(min, max), // our initial guess is just as bad
        Epanechnikov,                                           // Epanechnikov kernel for the rescue 
    );

    // Run 100 trials for the cosine function and try to find the point `(π, -1)`:
    let mut rng = Rng::new();
    let mut x = OrderedFloat(f64::NAN);
    for _ in 0..100 {
        x = optimizer.new_trial(&mut rng);
        println!("x = {x}");
        optimizer.feed_back(x, OrderedFloat(x.cos()));
    }
    println!("optimizer = {optimizer:?}");
    assert_abs_diff_eq!(optimizer.best_trial().unwrap().parameter.into_inner(), PI);
}
```
