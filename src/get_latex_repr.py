# %%
import enum
import re
from torch import nn
from torchinfo import summary
from pathlib import Path
import pandas as pd
from models.neural_network_VGG import (
    NeuralNetwork,
    NeuralNetwork_local,
    NeuralNetwork_server,
)

# %%
def get_latex_repr(model, input, **kwargs):

    out = summary(model, depth=2, input_size=input, **kwargs)

    regex = re.compile(
        r"^(├─|\│\s+)(.+)(\[\d+\,\s?\d+(\,\s?\d+)?(, \s?\d+)?\])(.*)", re.M
    )

    matches = regex.findall(str(out))

    df = pd.DataFrame(matches)
    df = df.drop([3, 4], axis=1)
    df[0] = df.apply(
        lambda x: x[1].strip() if x[0] == df[0].unique()[0] else None, axis=1
    )
    df[0] = df[0].ffill()
    df[1] = df[1].str.replace("└─", "").str.strip()

    df.columns = ["Module", "Layer Type", "Input Shape", "# Parameters"]
    df = df.set_index(["Module", "Layer Type"])
    latex_str = df.to_latex()
    return latex_str


# %%
output = Path("../tables")
output.mkdir(parents=True, exist_ok=True)


# %%

model = NeuralNetwork()
input_lastLayer = model.classifier[6].in_features
model.classifier[6] = nn.Linear(input_lastLayer, 10)

# %%
latex_str = get_latex_repr(model, (64, 3, 32, 32))
with open(output / "whole_vgg.txt", "w") as fh:
    fh.write(latex_str)

# %%
compressionProps = {}  ###
compressionProps[
    "feature_compression_factor"
] = 1  ### resolution compression factor, compress by how many times
compressionProps[
    "resolution_compression_factor"
] = 1  ###layer compression factor, reduce by how many times TBD

model_local = NeuralNetwork_local(compressionProps)

latex_local = get_latex_repr(model_local, (64, 3, 32, 32), local=True)

with open(output / "local_vgg.txt", "w") as fh:
    fh.write(latex_local)


# %%
model_server = NeuralNetwork_server(compressionProps)

latex_server = get_latex_repr(model_server, (64, 3, 32, 32), local=False)

with open(output / "server_vgg.txt", "w") as fh:
    fh.write(latex_server)
# %%

# %%
