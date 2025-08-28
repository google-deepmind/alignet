# AligNet Project Training Code, Data, and Model Checkpoints

This repository contains code and dataset information for ["Aligning Machine and Human Visual Representations across Abstraction Levels."](https://arxiv.org/abs/2409.06509)
Specifically, it includes the code for finetuning a pretrained SigLIP model on
the AligNet dataset, as well as links and documentation for the dataset and
the aligned model checkpoints.

Quick links:

* [Installation](#installation)
* [AligNet dataset](#alignet-dataset)
* [Run AligNet finetuning on SigLIP](#run-alignet-finetuning-on-siglip)
* [Released AligNet models](#alignet-models)
* [Citation](#citation)
* [License](#license)

## Motivation

Alignment with human mental representations is becoming central to
representation learning: we want neural network models that perform well on
downstream tasks **and** align with the hierarchical nature of human semantic
cognition. We believe that aligning neural network representations with human
conceptual knowledge will lead to models that generalize better, are more
robust, safer, and practically more useful. To obtain such models, we generated
a synthetic human-like similarity judgment dataset on a much larger scale than
has previously been possible. We have released this dataset, example finetuning
code for using it, and some finetuned versions of prior models.

Please see the [AligNet paper](https://arxiv.org/pdf/2409.06509) for further
details on the motivation and procedures.

## Installation

* Clone Repository

  ``` bash
  git clone https://github.com/google-deepmind/alignet.git
  ```
* Install requirements

  ```bash
  pip install -r alignet/requirments.txt
  ```

## AligNet dataset

The AligNet dataset is a synthetically generated dataset of image triplets
(sampled from ImageNet2012) and corresponding human-like triplet odd-one-out
choices.

### AligNet triplets
Download the data from https://console.cloud.google.com/storage/browser/alignet

AligNet is a dataset of triplets and corresponding odd-one-out choices.
Each triplet contains 3 image filenames (the images are sampled from ImageNet)
and the predicted similarity between those three images (obtained from a
pre-trained neural network).

To increase the reproducibility of our research, we split AligNet into a
training and a validation set. The train split `alignet_train.npz` contains 10M
triplets and the validation split `alignet_valid.npz` contains 10k triplets.
The files are stored in
[Numpy’s compressed array format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html).
Each file contains three arrays of *n* entries each, where `n=10M` for training
and `n=10k` for validation. Row *i* describes the *i*th triplet. Note that
within each triplet we sorted the images such that the last image is always the
one that is most dissimilar to the other two (i.e., the "odd-one-out"),
according to a prediction made by a model we trained (see the
[AligNet paper](https://arxiv.org/pdf/2409.06509) for details).

* `filenames`: (n, 3) strings: Identifies the images used for this triplet.
  Each row contains the names of image files from the ImageNet2012 dataset as
  [filename0, filename1, filename2], where filename2 is the image that is
  typically considered the "odd one out" of the triplet.
* `similarities`: (n, 3) floats: the similarity values of the three pairs of
  images calculated using the pretrained model representations:
  `[s01, s02, s12]`, where `sij` is the similarity between image i and image j.
   Note that the data was sorted such that `s12 < s01` and `s12 < s02`.
* `indices`: The indices of the three images in a
  [`tfds.datasource`](https://www.tensorflow.org/datasets/api_docs/python/tfds/data_source)
  of `imagenet2012`. Note that this matrix is redundant and allows easier access
  to the data (without going through the filenames) when using tfds.

### Supplementary Information

In addition to the official AligNet dataset, we provide other versions of the
data that we think might be useful to the broader community for running
additional experiments. These are other versions of the AligNet data that we
used for ablation studies. Concretely, we used variants that contain the
penultimate (or embedding) layer activations using:

* 3 different ways to sample triplets, depending on the imagenet labels of the
  images in the class:
    * *between-class*: all 3 images correspond to three different classes (this
    is similar to vanilla random sampling)
    * *class-border*: 2 images are sampled (without replacement) from the same
    class and one from a different
    * *within-class*: all images in a triplet are sampled (without replacement)
    from the same class

#### Representations

For each ImageNet image, we include the last-layer activations of the
open-source foundation model (so400m-siglip-webli384) used to compute the
triplet similarities, as well as the cluster assignments obtained from
clustering these activations into 500 clusters using k-Means.

### ImageNet
AligNet training depends on the tensorflow_datasets
[`imagenet2012`](https://www.tensorflow.org/datasets/catalog/imagenet2012)
dataset in `array_record` format.

This dataset requires you to download the source data manually into
`download_config.manual_dir` (defaults to
`~/tensorflow_datasets/downloads/manual/`):
`manual_dir` should contain two files: `ILSVRC2012_img_train.tar` and
`ILSVRC2012_img_val.tar`. You need to register on
https://image-net.org/download-images in order to get the link to download the
dataset.

After downloading the files (approx 150GB) the creation of the dataset can be
triggered by running (takes about an hour):

```python
import tensorflow_datasets as tfds
ds = tfds.data_source("imagenet2012", split="train")
```

NOTE: Note the use of `tfds.data_source` rather than `tfds.load`. This is needed
 because otherwise TFDS defaults to generating the dataset in TFRecords format
 which doesn't support random access, and does not work with AligNet finetuning.

## Levels dataset
As part of the AligNet project we also collected an evaluation dataset of human
 similarity judgments spanning multiple levels of semantic abstraction.
It can be found here: https://gin.g-node.org/fborn/Dataset_Levels

## Run AligNet finetuning on SigLIP

* Navigate to the parent directory of the `alignet` repository.
* Adjust `--cfg.aux.data_dir` to point to the directory containing the AligNet
  triplets.
* Point `--cfg.workdir` to the directory to which checkpoints etc should be
  saved.
* Run:

  ```bash
  python -m kauldron.main \
    --cfg=alignet/configs/siglip.py \
    --cfg.workdir=/tmp/kauldron/workdir \
    --cfg.aux.data_dir=/path/to/alignet/dataset
  ```

## AligNet models

We have exported AligNet post-trained versions of several models, which are
available at https://console.cloud.google.com/storage/browser/alignet/models

The models are released in the Tensorflow
[SavedModel](https://www.tensorflow.org/guide/saved_model) format.
We provide 8 different models:

* SigLIP-B
* SigLIP2-B
* DINOv1-B
* DINOv2-B
* CapPa-B
* ViT-B
* CLIP-Vit-B
* Scratch-B

For each model we provide three variants:
 * `MODEL-base_model`: The pre-trained base model before any AligNet post-training.
 * `MODEL-alignet`: The AligNet post-trained model.
 * `MODEL-untransformed`: The UnAligNet post-trained model.

Each model comes as a separate `.tar.gz` file that needs to be downloaded and
extracted. Then it can then be loaded and run as follows:

```python
import tensorflow as tf
import numpy as np

MODEL_NAME = "SigLIP-B-alignet"  # name of the model directory

images = np.zeros((8, 224, 224, 3), dtype=np.float32) # f32[B H W C]

m = tf.saved_model.load(MODEL_NAME)
forward = m.signatures['serving_default']
output = forward(images=images)
```

The `output` is a dictionary with the following entries:

  * `'pre_logits': f32[B H]` The logits of the layer before the readout heads. The dimension `H` varies between models (768-1536)
  * `'i1k_logits' : f32[B 1000]` The logits of the ImageNet2012 readout head.
  * `'triplet_logits': f32[B 1024]` The logits of the triplet head used during the AligNet post-training.
  * `'layer_{NUM}': f32[B 196 H]` Corresponds to the internal representations (14*14 = 196 tokens) after each of the (typically 12) layers.

## Citation

If you use the models, code, or dataset, we would appreciate if you could cite
the corresponding paper as follows:

```
@article{muttenthaler2024aligning,
  title={Aligning Machine and Human Visual Representations across Abstraction Levels},
  author={Muttenthaler, Lukas and Greff, Klaus and Born, Frieda and Spitzer, Bernhard and Kornblith, Simon and Mozer, Michael C and M{\"u}ller, Klaus-Robert and Unterthiner, Thomas and Lampinen, Andrew K},
  journal={arXiv preprint arXiv:2409.06509},
  year={2024}
}
```

## Dataset Metadata

The following table is necessary for this dataset to be indexed by search
engines such as [Google Dataset Search](https://g.co/datasetsearch).
<div itemscope itemtype="http://schema.org/Dataset">
<table>
  <tr>
    <th>property</th>
    <th>value</th>
  </tr>
  <tr>
    <td>name</td>
    <td><code itemprop="name">AligNet Dataset</code></td>
  </tr>
  <tr>
    <td>url</td>
    <td><code itemprop="url">https://github.com/google-deepmind/alignet</code></td>
  </tr>
  <tr>
    <td>sameAs</td>
    <td><code itemprop="sameAs">https://github.com/google-deepmind/alignet</code></td>
  </tr>
  <tr>
    <td>description</td>
    <td><code itemprop="description">
      A dataset of synthetic Human Preference Triplets based on ImageNet.
      </code></td>
  </tr>
  <tr>
    <td>provider</td>
    <td>
      <div itemscope itemtype="http://schema.org/Organization" itemprop="provider">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">DeepMind</code></td>
          </tr>
          <tr>
            <td>sameAs</td>
            <td><code itemprop="sameAs">https://en.wikipedia.org/wiki/DeepMind</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>citation</td>
    <td><code itemprop="citation">Muttenthaler L, Greff K, Born F, Spitzer B, Kornblith S, Mozer MC, M&uuml;ller KR, Unterthiner T, Lampinen AK (2024). Aligning machine and human visual representations across abstraction levels. arXiv preprint arXiv:2409.06509.</code></td>
  </tr>
</table>
</div>

## License

The AligNet dataset is under the
[CC-BY License](https://creativecommons.org/licenses/by/4.0/), and the
accompanying code is provided under an
[Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
Other parts of the datasets are under the original license of their sub-parts.
The aligned model checkpoints are governed by their original licenses; license
information is provided along with the checkpoints.

This is not an officially supported Google product.
