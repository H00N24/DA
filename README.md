## Domain Adaptation framework

This repo will help you to seamlessly adapt a language model on your own data.

### How to do it:

1. Find a data of your desired domain.
2. Pick **objectives**: pre-training and fine-tuning tasks to train your model on, 
and give the objectives a reference to their training texts.
3. Pick a **schedule** in which to apply the objectives. For starters, `SequentialObjective` 
will apply the objectives sequentially in the order that you'd give it.
4. Run an adaptation: wait for the schedule to run over the objectives, 
until the given terminating condition apply.
5. Save the model with all its heads and load it back 
after the training in the same way as before (whatever way that is).



```shell
git clone {this repo}
cd DA
python -m pip install -r

# calm down, we'll make a package out of this soon
export export PYTHONPATH="${PYTHONPATH}:$(pwd)/domain_adaptation
```

Then you can run the examples in `tests` folder. See for example `tests/end2end_usecases_test.py`