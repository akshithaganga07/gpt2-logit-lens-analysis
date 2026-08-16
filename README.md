# GPT-2 Logit Lens Analysis

## Overview

This project investigates how factual predictions emerge across the transformer layers of GPT-2 using the **Logit Lens** interpretability technique.

Instead of examining only the final output of GPT-2, this project projects intermediate hidden representations into the model's vocabulary space. This makes it possible to observe which tokens become more probable at different layers.

## Objectives

* Analyze intermediate representations of GPT-2.
* Implement Logit Lens using Python and PyTorch.
* Track expected-answer token probabilities across GPT-2 layers.
* Compare layer-wise prediction behavior across multiple factual prompts.
* Visualize how predictions emerge throughout the model.
* Explore the strengths and limitations of Logit Lens for language-model interpretability.

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* GPT-2
* Pandas
* Matplotlib
* Google Colab

## Methodology

The experiment follows these steps:

1. Load the GPT-2 pretrained language model.
2. Provide factual prompts to the model.
3. Extract hidden representations from each transformer layer.
4. Project intermediate representations through GPT-2's language-model head.
5. Calculate vocabulary-token probabilities.
6. Track the probability of expected answer tokens across layers.
7. Compare the results across multiple prompts.
8. Visualize the layer-wise probability changes.

## Experimental Prompts

The experiment uses five factual prompts:

| Prompt                                    | Expected Answer |
| ----------------------------------------- | --------------- |
| The capital of France is                  | Paris           |
| The capital of Japan is                   | Tokyo           |
| The largest planet in our solar system is | Jupiter         |
| Water freezes at                          | 0               |
| The Eiffel Tower is located in            | Paris           |

## Results

The experiment showed that expected-answer tokens generally became more prominent around the later GPT-2 layers, particularly layers 10–11.

| Prompt               | Peak Layer | Maximum Probability |
| -------------------- | ---------: | ------------------: |
| France → Paris       |         10 |              18.19% |
| Japan → Tokyo        |         11 |              46.11% |
| Planet → Jupiter     |         11 |              36.44% |
| Water → 0            |         11 |               6.08% |
| Eiffel Tower → Paris |         11 |               1.29% |

The strength of the expected-answer signal varied considerably between prompts. This suggests that different factual associations are represented with different strengths within the model.

## Key Finding

The analysis indicates that useful factual information can become visible in intermediate GPT-2 representations before the final output stage. However, the probability of an expected answer does not necessarily increase monotonically through the network.

This demonstrates how Logit Lens can provide a simple window into the evolution of model predictions across transformer layers.

## Limitations

* The experiment uses the relatively small GPT-2 model.
* Only five factual prompts were evaluated.
* The probability analysis focuses primarily on single-token answers.
* Logit Lens is a probing technique and should not be interpreted as a direct representation of the model's internal reasoning.
* Results can be sensitive to prompt wording.
* The experiment does not compare multiple model architectures or sizes.

## Future Work

* Compare different GPT-2 model sizes.
* Extend the analysis to multi-token answers.
* Evaluate a larger and more diverse prompt dataset.
* Compare Logit Lens with other interpretability techniques.
* Analyze attention patterns alongside hidden representations.
* Compare layer-wise behavior across different language models.

## Notebook

The complete implementation and experimental results are available in:

`GPT2_Logit_Lens_Analysis.ipynb`

The notebook can be run using **Google Colab**.

## Author

**Akshitha Ganga**

B.Tech — Electronics and Communication Engineering
