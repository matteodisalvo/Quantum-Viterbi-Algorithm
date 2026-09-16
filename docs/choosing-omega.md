# Choosing ω

The phase oracle writes the path metric into a phase, $G(\omega)\,|u\rangle = e^{i\omega\,d_H(u)}\,|u\rangle$,
so ω decides how well the amplification favors the maximum-likelihood path. There is no
single good value: it depends on the received word, and a bad ω amplifies the wrong path.
The workflow is therefore to tune ω for that word and then decode with it. See also
[Theory](theory.md) and the [README](../README.md).

## What `decode` does

Called without ω, `decode` tunes it with `tuned_omega`. `decode` reports the single most
probable path, so what matters is not only how probable a maximum-likelihood path is, but
also whether some wrong path is more probable still. With $p$ the probability of one
maximum-likelihood path and $m$ its lead over the most probable path at any other
distance, `tuned_omega` maximizes

$$\text{score}(\omega) = \begin{cases} \sqrt{p\,m} & m > 0 \\ m & m \le 0 \end{cases}$$

An ω that decodes to a wrong message ($m \le 0$) never beats one that decodes correctly,
and a tie scores zero. A coarse grid over $(0, \pi]$ locates the peaks of the score and the
five highest are refined by re-gridding the bracket around the best point.

```python
from quantum_viterbi import decode, tuned_omega

received = "11 01 00 10 00 00"
tuned_omega(received)  # ≈ 0.6819
decode(received)       # same value, reported as "omega : 0.6819 rad (tuned)"
```

Why not maximize $p$ alone? For an *aliased* word, one with paths at distance
$2N - d_{\min}$, those paths reach almost the same phase as the maximum-likelihood one and
are amplified with it, so the most probable path can end up being a wrong one, or tie with
the right one. Over all 1024 received words with $N = 5$, maximizing $p$ decodes 384 of
them wrongly or on a tie; the score above decodes all of them correctly. For 16 of the 256
words with $N = 4$ no ω decodes correctly at the default $K = 3$, while a single Grover
round does.

All paths at the same distance $d$ keep the same amplitude, so the score is
evaluated on the $2N+1$ distance classes (see below) instead of the $2^N$ statevector:
the whole search costs $O(\text{points} \cdot K \cdot N)$. It takes a few hundredths of a
second up to $N = 12$, about a fifth of a second at $N = 16$ and about two seconds at
$N = 20$, where the grid itself has to grow.

The peaks get narrower as $N$ grows, so the coarse grid grows with $N$ up to 50001 points.
Up to $N = 16$ the score reached matches the best score of a grid ten times finer;
above that a grid can step over a peak entirely, and the resonance below locates it
analytically.

## Grid search

`scan_omega` is the method used for the experiments of the thesis: evaluate the
probability of a target path on a grid of ω values and keep the best one. It maximizes
that probability alone, so its ω can differ slightly from the one of `tuned_omega`.

```python
from quantum_viterbi import omega_range, scan_omega, viterbi_decode

target = viterbi_decode(received).bits          # the classical maximum-likelihood path

grid = omega_range(0, 180, 0.01, degrees=True)  # [0, pi] is enough: P is 2*pi-periodic
scan = scan_omega(received, grid, target=target)
print(scan.best_omega, scan.best_score)         # ≈ 0.6733  ≈ 0.2971
```

Use a target path. Maximizing only the peak probability can pick a wrong one: with
ω ≈ 0.565 the most probable outcome of this word is `001001` (P ≈ 0.44), 11 bit errors
away from the received word. The mean path distance is always $N = 6$, and that path lies
as far above it (11 − 6) as the maximum-likelihood path lies below it (6 − 1), so near
ω ≈ π/5 both get a phase opposite to the bulk of the paths.

## Resonant ω

The distance classes also give ω in closed form. With weights $w_d = A_d/2^N$, where $A_d$
is the number of paths at distance $d$ (computed by a trellis pass in $O(N^2)$), for the
maximum-likelihood distance $d_0 = d_{\min}$ the search behaves approximately like a
detuned Grover search, and a good ω is a root of

$$C(\omega) = \sum_{d \neq d_0} w_d \cot\frac{\omega\,(d-d_0)}{2} = 0$$

At such a root the other classes balance out, the target sees an undetuned Grover search,
and the two-level model predicts both the peak probability $1/S$ and the number of
iterations

$$S(\omega) = \sum_{d \neq d_0} \frac{w_d}{\sin^2\big(\omega\,(d-d_0)/2\big)}, \qquad K^\ast = \frac{\pi}{4}\sqrt{\frac{S}{w_0}} - \frac{1}{2}$$

```python
from quantum_viterbi import decode, resonant_omega, scan_resonance

res = resonant_omega(received)    # omega, iterations, predicted probability, width
print(res.omega, res.iterations)  # ≈ 0.6677  9

result = decode(received, omega=res.omega, iterations=res.iterations)
print(result.best_probability)    # ≈ 0.4547

scan = scan_resonance(received)   # exact scan around the resonance, default iterations
print(scan.best_omega, scan.best_score)  # ≈ 0.673  ≈ 0.2971
```

The roots lie near $\pi/(N - d_{\min})$, the antipodal condition of Grice and Meyer
($\pi/N$ for noiseless words), which is why their tabulated values are so close to $\pi/N$.
That approximation should not be used directly, though: in the example above the path at
distance $2N - d_{\min} = 11$ gets exactly the target phase at ω = π/5, and both end up
with P ≈ 0.24.

Exact probabilities on the distance classes are available on their own:

```python
from quantum_viterbi import class_probabilities, distance_spectrum

distance_spectrum("11 01 00 10 00 00")            # (0, 1, 2, 4, 6, 9, 14, 15, 10, 2, 0, 1, 0)
class_probabilities("00" * 20, [0.157133], 834)   # exact P per distance at N = 20
```

## What to expect

On the word above, at the default six Grover iterations:

| ω                     | Where it comes from                   | Value  | P(101101) | Lead    | Decodes 101101 |
|-----------------------|---------------------------------------|--------|-----------|---------|----------------|
| `tuned_omega`         | the score above, on the received word | 0.6819 | 0.2931    | +0.2007 | yes            |
| `scan_omega`          | highest P(101101) on an explicit grid | 0.6733 | 0.2971    | +0.1900 | yes            |
| `resonant_omega`      | the distance spectrum                 | 0.6677 | 0.2956    | +0.1774 | yes            |
| `channel_omega(N, p)` | only $N$ and the channel              | 0.6093 | 0.2223    | -0.0961 | no             |

The lead is how much more probable 101101 is than the most probable wrong path, and
`decode` returns the right message only when it is positive. `scan_omega` reaches the
highest probability but a smaller lead than `tuned_omega`. `resonant_omega` also suggests
its own number of iterations, $K = 9$ here, which raises the probability to 0.4547 and the
lead to 0.3603. `channel_omega` is the only one that never looks at the word, and it pays
for it: on this word a wrong path is more probable than the right one.

Two caveats:

- **Tuning uses the received word.** Its distance spectrum already reveals $d_{\min}$,
  which is what the classical Viterbi algorithm computes, so `tuned_omega` and
  `resonant_omega` are tools for analysis and benchmarking, not free information for a
  decoder.
- **The resonance is narrow.** Its width shrinks roughly as $4 \cdot 2^{-N/2}/N$ (about
  $2 \cdot 10^{-4}$ rad at $N = 20$), so ω must be set very precisely, both in a
  simulation and, more seriously, on hardware.
