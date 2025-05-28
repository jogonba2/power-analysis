# ⚡ Power
<p align="center">
    <a href="LICENSE">
        <img alt="license" src="https://img.shields.io/badge/license-Apache_2.0-green">
    </a>
</p>

<h3 align="center">
    <p><b>Supporting NLP researchers in making statistically informed evaluation decisions</b></p>
</h3>

# 📖 What is this repo for?
This repository contains the source code of a simulator to compute **statistical power** of evaluation experiments, that can help researchers addressing **post-hoc** and **pre-hoc** analyses.

In post-hoc analyses, researchers have evaluated two models, $A$ and $B$, on a test set and compared their performances, ideally considering a statistical test to reject the null hypothesis defined as $H_0: A=B$. Let's suppose that the null hypothesis has been rejected in that experiment. Then, the following question arises: what is the probability of **correctly** rejecting that false null hypothesis with your experiment? This is the statistical power of your experiment, and it is crucial to ensure your experiments have enough power: underpowered studies not only fail to detect real effects but actively distort the results that achieve statistical significance. This repo will help you in determining it such post-hoc analysis.

In pre-hoc analyses, researchers would like to know how many samples are required to have enough statistical power in model comparisons, before building a dataset. This way, we can collect enough test samples to determine later that two models are statistically different with enough power.

# 🎭 Post-hoc analysis in a text classification task

This example will guide you to understand how to compute the statistical power of your experiment to compare two sentiment analysis models, $A$ and $B$, trained with the IMDB dataset.

We provide a tour for practitioners, showing how the statistical power can be computed with a single line, and a guide for researchers to understand the internals.

## 👨‍🏭 Practitioner tour 

The statistical power can be computed just calling `classification_report` by providing the predictions of your models and the reference labels.

Let's first load the dataset, the trained models, and predict the test set:

```python
from transformers import pipeline
from datasets import load_dataset
import numpy as np

# Load the test set and keep it small for the example
test_set = load_dataset("stanfordnlp/imdb", split="test")
test_set = test_set.select(range(100))

# Create the pipelines for inference
model_A = pipeline(model="lvwerra/distilbert-imdb", batch_size=32, device=0)
model_B = pipeline(model="jialicheng/deberta-base-imdb", batch_size=32, device=0)

# Unify the label dicts for the example
model_A.model.config.id2label={0: "neg", 1: "pos"}
model_B.model.config.id2label=model_A.model.config.id2label

# Predict with the pipelines
preds_A = model_A(test_set["text"], truncation=True)
preds_B = model_B(test_set["text"], truncation=True)

# Gather predictions as numpy arrays
preds_A = [pred["label"] for pred in preds_A]
preds_B = [pred["label"] for pred in preds_B]
references = test_set.features["label"].int2str(test_set["label"])
```

Then, we only have to call `classification_report` by passing the model predictions and references, the number of iterations for the simulation, the significance level, and a random seed for reproducibility. Users can also specify the data generation process, the statistical test, and the effect size function, as long as these are provided by `power`. By default, these are configured to the most suited ones for classification tasks. If you want to go deeper into these topics, please, read the tour for researchers.

```python
from power.metrics import classification_report

power = classification_report(
    preds_A, preds_B, references, iterations=1000, alpha=0.05, seed=13
)
```

Nice! You have computed the power of your experiment, together with other useful metrics like [Type S and M errors](https://statmodeling.stat.columbia.edu/2004/12/29/type_1_type_2_t/) and mean effect.

> [!TIP]
> After running the code above, the power should be around 17.8%. As convention, ≥ 80% power can be considered enough to ensure that, in your experiment, you rejected the null hypothesis correctly. Otherwise, your experiment can be considered underpowered, i.e., it cannot provide useful evidence that one model achieves slightly better performance than another for the underlying data distribution.

## 👩‍🔬 Researcher tour 

Here, we show the whole flow a researcher can follow to compute the statistical power, by replicating the example in the practitioner tour but with access to the internals. This allows you to change any component of the simulation, including: data generation process (DGP), statistical test, and effect functions.

As in the tour for practitioners, the first step is to load the dataset, the trained models, and predict the test set:

```python
from transformers import pipeline
from datasets import load_dataset
import numpy as np

# Load the test set and keep it smaller for the example
test_set = load_dataset("stanfordnlp/imdb", split="test")
test_set = test_set.select(range(100))

# Create the pipelines for inference
model_A = pipeline(model="lvwerra/distilbert-imdb", batch_size=32, device=0)
model_B = pipeline(model="jialicheng/deberta-base-imdb", batch_size=32, device=0)

# Unify the label dicts for the example
model_A.model.config.id2label={0: "neg", 1: "pos"}
model_B.model.config.id2label=model_A.model.config.id2label

# Predict with the pipelines
preds_A = model_A(test_set["text"], truncation=True)
preds_B = model_B(test_set["text"], truncation=True)

# Gather predictions as numpy arrays
preds_A = np.array([pred["label"] for pred in preds_A])
preds_B = np.array([pred["label"] for pred in preds_B])
references = np.array(test_set.features["label"].int2str(test_set["label"]))
```

Nice. At this point we have the predictions of models $A$ and $B$, together with the reference labels in the test set. Now we can start preparing the flow to compute the power. First, we need to create a 2x2 contingency table representing true conditional probabilities. This matrix looks like:

|               | $B$ correct                      | $B$ incorrect                    |
|---------------|----------------------------------|----------------------------------|
| $A$ **correct**   | p($A$ and $B$ correct)           | p($A$ correct and $B$ incorrect) |
| $A$ **incorrect** | p($A$ incorrect and $B$ correct) | p($A$ and $B$ incorrect)         |

So, let's compute these values and store them in the `true_prob_table` variable:

```python
# Prepare the true probability table
dataset_size = len(references)

true_prob_table = np.array([
    [
        ((preds_a != references) & (preds_b != references)).mean(),
        ((preds_a == references) & (preds_b != references)).mean()
    ],
    [
        ((preds_a != references) & (preds_b == references)).mean(),
        ((preds_a == references) & (preds_b == references)).mean()
    ]
])
```

Now, we have all the information required to instantiate the generation process $\mathcal{G}$ that will be used to sample synthetic data from `true_prob_table`. Since we will use the [McNemar test](https://en.wikipedia.org/wiki/McNemar%27s_test) in this experiment as statistical test to compute a p-value and reject/accept the null hypothesis ($H_0: A=B$), here we will use `contingency_table` as DGP. This DGP is already provided by the `power` package and it will sample contingency tables from the true probability table we have computed, to be used later as inputs to the statistical test. Nothing prevents you to create other DGP functions!

> [!NOTE]  
> In this context, synthetic data refers to data sampled from the `true_prob_table`. This is purely statistical based data generation, not related in any case with synthetic data as commonly understood in NLP with techniques like backtranslation, or LLM generation.

Let's instantiate the generation process function $\mathcal{G}$:

```python
from power.dgps import dgps
from power.types import DGPParameters

dgp_args = DGPParameters(
    true_prob_table=true_prob_table, dataset_size=dataset_size
)

data_generating_fn = partial(
    dgps.get("dgp::contingency_table"), dgp_args=dgp_args
)
```

Perfect, we now have to instantiate the statistical test $\mathcal{T}$ we want to use and the estimated effect function $\mathcal{E}$. As statistical test, we will use McNemar's test, and the estimated effect will measure the difference between both models focusing on the difference of discordant pairs ($A$ correct & $B$ incorrect and $A$ incorrect & $B$ correct). Luckily, `power` already provides us implementations both for the statistical test and estimated effect: `mcnemar` and `cohens_g`. Again, nothing prevents you for implementing your own statistical tests and functions to estimate the effect size. Let's instantiate `mcnemar` and `cohens_g` for this example:

```python
from power.effects import effects
from power.stats_tests import stats_tests
from functools import partial

# Retrieve the function to compute Cohen's g as the effect size
effect_fn = effects.get("effect::cohens_g")

# Create a function for the McNemar test
statistical_test_fn = partial(
    stats_tests.get("stats_test::mcnemar"), effect_fn=effect_fn
)
```

In this way, the estimated effect size will be computed on the synthetic sample and McNemar will return both the $p$-value and the estimated effect size. But we need the **true** effect size $\delta^*$ too to run the simulation. This true effect size is computed with the `true_prob_table` instead of with a synthetic sample. So, let's define our true effect function:

```python
# Prepare the function to compute the *true* effect size for benchmarking
true_effect_fn = partial(
    effect_fn,
    true_prob_table=true_prob_table,
    sample=None,
    dataset_size=dataset_size,
)
```

Now we have all the functions required to run the simulation to compute the statistical power! To run the simulation, `power` provides the function `compute_power` that receives as arguments the data generation process function $\mathcal{G}$, the statistical test $\mathcal{E}$, the true effect function that computes the true effect $\delta^*$, the number of iterations $S$, the significance level $\alpha\in(0,1)$, and a random seed to ensure reproducibility. `compute_power` returns the power of your experiment, so let's call it:

```python
from power.compute_power import compute_power 

power = compute_power(
    data_generating_fn,
    statistical_test_fn,
    true_effect_fn,
    iterations=1000,
    alpha=0.05,
    seed=13,
)
```

Congrats! You have already computed the power of your experiment by using the internals of `power`.

> [!NOTE]  
> Through this example, we shown how to use existing functionality in the `power` package to compute statistical power in a specific scenario: text classification task, using the McNemar test as statistical test, and measuring the effect size focusing on discordant pairs. If this does not fit your scenario, you can use the registers provided by `power` to register any additional function you need.