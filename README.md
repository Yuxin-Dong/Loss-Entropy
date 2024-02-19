# Rethinking Information-theoretic Generalization: Loss Entropy Induced PAC Bounds

Our supplementary material enables the replication of two experiments:
* Correlation Analysis
* Bound Comparison

## Correlation Analysis

The code is developed based on [this repository](https://github.com/xu-ji/information-bottleneck).

The analysis of $H(L^w)$ and $H(L^W|Y)$ is implemented in `information-bottleneck/toy/scripts/main_swag.py`, with the kernel density estimator implemented in `information-bottleneck/toy/util/general.py`.

## Bound Comparison

The code is developed based on [this repository](https://github.com/hrayrhar/f-CMI).

The estimation of our bounds is implemented in `f-CMI/scripts/fcmi_parse_results.py` and `f-CMI/modules/bound_utils.py`. The figures are plotted using the code in `f-CMI/scripts/mee_plots.py`.

## Cite

```
@inproceedings{
	dong2024rethinking,
	title={Rethinking Information-theoretic Generalization: Loss Entropy Induced {PAC} Bounds},
	author={Yuxin Dong and Tieliang Gong and Hong Chen and Shujian Yu and Chen Li},
	booktitle={The Twelfth International Conference on Learning Representations},
	year={2024},
	url={https://openreview.net/forum?id=GWSIo2MzuH}
}
```