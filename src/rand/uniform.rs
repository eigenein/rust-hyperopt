/// Uniform distribution.
pub trait Uniform<T> {
    fn uniform(&mut self) -> T;
}

impl Uniform<bool> for fastrand::Rng {
    /// Generate a random boolean value with 50% probability of each class.
    #[inline]
    fn uniform(&mut self) -> bool {
        self.bool()
    }
}

impl Uniform<f64> for fastrand::Rng {
    /// Generate a uniformly random [`f64`] in range `0..1`.
    #[inline]
    fn uniform(&mut self) -> f64 {
        self.f64()
    }
}

impl Uniform<f32> for fastrand::Rng {
    /// Generate a uniformly random [`f32`] in range `0..1`.
    #[inline]
    fn uniform(&mut self) -> f32 {
        self.f32()
    }
}
