# Experimental Observations

## StackCube

With matched 7,500-step training and P8+A10 rollout:

| Cache | seed42 | seed43 | Total |
|---|---:|---:|---:|
| stride=8 | 42/50 | 44/50 | 86/100 |
| full cache | 37/50 | 41/50 | 78/100 |

The seed43 full-cache rerun varied from an earlier 44/50 result, so the observed advantage is a favorable trend rather than a definitive significance claim.

## CloseBox

| Configuration | Result |
|---|---:|
| stride=8, P8+A10, 7,500 updates | 15/50 |
| FM baseline A10 | 15/50 |
| stride=8, P8+A10, 37,500 updates | 30/50 |

CloseBox shows that a temporal-thinning cache still needs sufficient optimization. The 7,500-step run did not improve over the baseline; the 37,500-step run did.

Results use fixed task, scene, environment seed, and source-noise schedule per comparison. Independent reruns retain simulator variation and should be reported with additional seeds for final claims.
