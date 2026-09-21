# Temporal Thinning for Prior Flow

## Problem

An expert-inversion cache is built from overlapping action windows. With a horizon of 16, adjacent windows differ by one timestep and share fifteen actions. Uniform training over every window therefore assigns disproportionate probability to temporally dense local segments.

## Method

For each `(split, episode)` group, order cache rows by `sampler_index`, retain every eighth row, and always retain the final row. The final-row rule preserves terminal behavior even when the trajectory length is not divisible by eight.

The cache is a direct subset of the full cache: conditions, inversion latents, episode ids, and split assignments are copied without recomputing inversion. Only the sampling distribution changes.

## Fixed-update training

A smaller cache has fewer natural minibatches per epoch. To compare it with full-data training fairly, the stride trainer cycles independently shuffled passes through the selected rows until it reaches the requested number of batches. Thus the comparison holds optimizer updates fixed while changing temporal redundancy.

## Inference

The Prior Flow uses left-Euler integration from Gaussian noise. P8 uses eight Prior steps. The output enters the frozen Action Flow, which uses ten midpoint steps in A10.

## Claim boundary

Temporal thinning is a sample-efficiency hypothesis, not a claim that less data is universally better. It must be evaluated with matched optimizer budgets and repeated rollout seeds.
