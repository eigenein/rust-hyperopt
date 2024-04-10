/// Single Kernel Density Estimator component.
#[derive(Copy, Clone, Debug)]
pub struct Component<K, T> {
    /// Kernel function.
    pub kernel: K,

    /// Center of the corresponding kernel.
    pub location: T,

    /// Bandwidth of the corresponding kernel.
    pub bandwidth: T,
}
