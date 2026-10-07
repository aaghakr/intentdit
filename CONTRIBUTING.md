# Contributing

1. Create a focused branch and keep generated data outside Git.
2. Run `cd code && python -m unittest discover -s tests -v` before opening a change.
3. Do not commit datasets, checkpoints, user-study databases, rendered
   outputs, or machine-specific absolute paths.
4. Add a test for changes to losses, coordinate transforms, padding, masking,
   configuration resolution, or metric definitions.
5. Record the dataset split, resolved configuration, training seed, inference
   seed, checkpoint, and commit hash for every reported result.
