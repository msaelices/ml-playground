# ml-playground
Machine Learning Playground

`mlpg` stands for `Machine Learning Playground`

It is a hands-on repository for self-learning of machine learning algorithms by solving real problems using pytorch and other ML libraries.

## ML Problems

### MNIST

The MNIST database of handwritten digits, available from [this page](http://yann.lecun.com/exdb/mnist/).

### MLP implementation

You can see the `MNISTModel` model in the [models.py](./src/projects/mnist/models.py) module, with an example of usage in the [MNIST notebook](./src/projects/mnist/notebooks/mnist_mlp.ipynb).

### ViT implementation

You can see the `MNISTViTModel` model in the [models.py](./src/projects/mnist/models.py) module, with an example of usage in the [MNIST ViT notebook](./src/projects/mnist/notebooks/mnist_vit.ipynb).

### Playground

You can also try the generated model yourself by running the following command in the `src/projects/mnist` directory:

```bash
python canvas.py
```

## Running the tests

```bash
python -m pytest tests.py
```
