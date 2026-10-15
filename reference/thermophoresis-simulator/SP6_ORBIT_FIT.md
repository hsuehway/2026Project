# SP6 circular-orbit fit

This fit reproduces the principal SP6 circular mode in
`ThermophoreticLevitationSimulator.html`.

## Measured target

- Direction: clockwise in the reconstructed camera `x-y` plane
- Frequency: 1.9431555 Hz
- Period: 0.5146268 s
- Representative RMS orbit radius: 0.34169 mm
- Tangential speed: 4.17176 mm/s
- Representative burst duration: approximately 25 s

The simulator uses coordinates relative to the local trap center. Its `z = 0`
therefore represents the observed levitation height, not the bottom plate.

## Initial state

| Quantity | Value |
| --- | ---: |
| Tilt | 0 deg |
| Position `x0` | `(0.00034169, 0, 0)` m |
| Velocity `v0` | `(0, -0.004171763675, 0)` m/s |
| Body angular velocity `omega0` | `(0, 0, -12.20920623)` rad/s |

The negative angular velocity gives the measured clockwise winding. With `v0 =
0`, the same orbit is reached after a short transient because the fitted
translational relaxation time is about 0.0245 s.

## Fitted simulator conditions

These are normalized effective coefficients for the HTML model. They reproduce
the trajectory but are not an independent measurement of the particle tensors.

```text
M = 1
g = 9.81
I = diag(0.02, 0.02, 0.008)

alpha_r = 28.83350715
alpha_z = 0.5

Gamma = [[ 5,            0, -0.0008528632769],
         [ 0,            5, -0.08685789134  ],
         [-0.0008528632769, -0.08685789134, 5]]

Xi = diag(40.84967320, 40.84967320, 40.84967320)
Cr = diag(0.5, 0.5, 0.5)

Lambda = [[0, 0,  0],
          [0, 0,  0],
          [0, 0, -3.111418508]]
```

Use `t_end = 25 s` and `12,000` RK4 steps for the high-resolution run. The
existing 1,200-step setting gives effectively the same fitted frequency and
radius.

## Verification

The simulator equations were evaluated with the same RK4 update used by the
HTML file.

| RK4 steps over 25 s | Frequency | Mean radius | Radius SD |
| ---: | ---: | ---: | ---: |
| 1,200 | -1.9431541 Hz | 0.341724 mm | 0.001280 mm |
| 12,000 | -1.9431557 Hz | 0.341681 mm | 0.001400 mm |

The minus sign denotes clockwise winding. The target is 1.9431555 Hz and
0.34169 mm.

## Why the default settings do not orbit

With the default isotropic `Gamma` and `Xi`, orientation drops out of the
translational force law. Changing only tilt, position, or angular velocity can
produce a transient, but not a sustained circular center-of-mass orbit. The fit
requires both:

1. a nonzero torque eigenvalue (`Lambda33`) to maintain rotation; and
2. a small off-diagonal thermophoretic-force response (`Gamma23`, with the much
   smaller phase correction `Gamma13`) so the rotating body drives a rotating
   horizontal force.

The HTML simulator now includes an **SP6 fitted orbit** preset and exposes the
initial translational velocity that was previously hard-coded to zero.
