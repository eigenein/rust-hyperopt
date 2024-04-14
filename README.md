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
        min, max,                            // parameter search range
        Uniform::<NotNan<f64>> { min, max }, // our initial guess is just as bad
    );

    // Run 100 trials for the cosine function and try to find the point `(π, -1)`:
    let mut rng = Rng::new();
    for _ in 0..100 {
        // Generate new trials using Epanechnikov kernel:
        let x = optimizer.new_trial::<Epanechnikov<_>, NotNan<f64>>(&mut rng);
        optimizer.feed_back(x, NotNan::new(x.cos()).unwrap());
    }

    let best_trial = optimizer.best_trial().unwrap();
    assert_abs_diff_eq!(best_trial.parameter.into_inner(), PI, epsilon = 0.3);
    assert_abs_diff_eq!(best_trial.metric.into_inner(), -1.0, epsilon = 0.05);
}
```

### Discrete

![TODO](https://img.shields.io/badge/TODO-red?style=for-the-badge)
