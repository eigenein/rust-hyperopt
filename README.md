# `hyperopt`

Tree-of-Parzen-estimators hyperparameter optimization for Rust

[![Documentation](https://img.shields.io/docsrs/hyperopt?style=for-the-badge)
](https://docs.rs/hyperopt)
[![Check status](https://img.shields.io/github/actions/workflow/status/eigenein/rust-hyperopt/check.yaml?style=for-the-badge)]((https://github.com/eigenein/rust-hyperopt/actions/workflows/check.yaml))
[![Code coverage](https://img.shields.io/codecov/c/github/eigenein/rust-hyperopt?style=for-the-badge)
](https://app.codecov.io/gh/eigenein/rust-hyperopt)
![Maintenance](https://img.shields.io/maintenance/yes/2024?style=for-the-badge)

## Examples

### Continuous

```rust
use std::f64::consts::{FRAC_PI_2, PI};

use approx::assert_abs_diff_eq;
use fastrand::Rng;
use ordered_float::NotNan;

use hyperopt::Optimizer;
use hyperopt::kernel::continuous::Epanechnikov;
use hyperopt::kernel::universal::Uniform;

fn main() {
    let min = NotNan::new(FRAC_PI_2).unwrap();
    let max = NotNan::new(PI + FRAC_PI_2).unwrap();
    let mut optimizer = Optimizer::new(
        min..=max,                       // parameter search limits
        Uniform::with_bounds(min..=max), // our initial guess is just as bad
        Rng::with_seed(42),
    );

    // Run 100 trials for the cosine function and try to find the point `(π, -1)`:
    for _ in 0..50 {
        // Generate new trials using Epanechnikov kernel with `<NotNan<f64>>`
        // as both parameter and density:
        let x = optimizer.new_trial::<Epanechnikov<NotNan<f64>>>();
        
        // Tell the optimizer the result of evaluation:
        optimizer.feed_back(x, NotNan::new(x.cos()).unwrap());
    }

    let best_trial = optimizer.best_trial().unwrap();
    assert_abs_diff_eq!(best_trial.parameter.into_inner(), PI, epsilon = 0.02);
    assert_abs_diff_eq!(best_trial.metric.into_inner(), -1.0, epsilon = 0.01);
}
```

### Discrete

```rust
use fastrand::Rng;
use ordered_float::OrderedFloat;

use hyperopt::Optimizer;
use hyperopt::kernel::discrete::Binomial;
use hyperopt::kernel::universal::Uniform;

fn main() {
    let mut optimizer = Optimizer::new(
        -100..=100,
        Uniform::with_bounds(-100..=100),
        Rng::with_seed(42),
    );

    for _ in 0..30 {
        // Use the binomial kernel for `i32` as parameter
        // and `OrderedFloat<f64>` as density:
        let x = optimizer.new_trial::<Binomial<i32, OrderedFloat<f64>>>();
        
        // Optimize the parabola: https://www.wolframalpha.com/input?i=x%5E2+-+4x
        optimizer.feed_back(x, x * x - 4 * x);
    }

    let best_trial = optimizer.best_trial().unwrap();
    assert_eq!(best_trial.parameter, 2);
    assert_eq!(best_trial.metric, -4);
}
```

## Features

- `ordered-float` enables support for `OrderedFloat` and `NotNan` types
