# ml-playground
Machine Learning Playground

`mlpg` stands for `Machine Learning Playground`

It is a hands-on repository for self-learning of machine learning algorithms by solving real problems using pytorch and other ML libraries.

## ML Problems

### MNIST

The MNIST database of handwritten digits, available from [this page](http://yann.lecun.com/exdb/mnist/).

You can see the Pytorch model in the [model.py](./mlpg/nnetworks/mnist/model.py) module, with an example of usage in the [MNIST notebook](./mlpg/nnetworks/mnist/notebook.ipynb).

You can also try the generated model yourself by running the following command in the `mlpg/nnetworks/mnist` directory:

```bash
python canvas.py
```

## Running the tests

```bash
python -m pytest tests.py
```
