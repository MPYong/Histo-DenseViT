# Histo-DenseViT

PyTorch implementation of "Histo-DenseViT: Combining Multi-scale Features in Hybrid Vision Transformer through Dense Connection for Histopathology Image Classification"

![alt text](https://raw.githubusercontent.com/MPYong/Histo-DenseViT/refs/heads/main/image/Graphical%20abstract.jpg?token=GHSAT0AAAAAACYIGRN7DKK4VADHCDLDL4V4ZYDWEJA)

The model configuration used in the paper is as following:

```
from histo_densevit import histo_densevit

model = histo_densevit \
        (image_size = (224,224), num_classes = 2,
        contextual_rpe = True,      # If True, use relative RPE, else use bias RPE
        contextual_input = 'q,k',   # Mode of relative RPE, 'q,k' indicates the relative RPE will be encoded
                                    # into queries and keys
        T_in_T = True,              # If True, deploys self-attention between windows
        deep_dense = True,          # If True, deploys dense connection
        )
```
