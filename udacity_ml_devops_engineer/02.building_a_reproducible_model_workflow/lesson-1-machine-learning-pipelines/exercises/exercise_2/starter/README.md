

- `conda.yml` specifies dependencies
- `mlproject` used `conda.yml` to find out the environment it should use
- it then runs the command to run the python script, passing the arguments using `argparse`
- the script downloaded the passed file URL
- `wandb` makes an artifact and uploads it.


#### command to run
- `./` the directory where the `mlproject` and `conda.yml` files are
- need to use `-P` before every passed arg

```powershell
MLFlow run ./  -P file_url="https://raw.githubusercontent.com/scikit-learn/scikit-learn/4dfdfb4e1bb3719628753a4ece995a1b2fa5312a/sklearn/datasets/data/iris.csv" -P artifact_name="iris" -P artifact_type="raw_data" -P artifact_description="The sklearn IRIS dataset"
```