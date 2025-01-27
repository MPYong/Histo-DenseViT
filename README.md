# Histo-DenseViT

PyTorch implementation of "Histo-DenseViT: Combining Multi-scale Features in Hybrid Vision Transformer through Dense Connection for Histopathology Image Classification"

![alt text](https://github.com/MPYong/Histo-DenseViT/blob/main/image/Graphical%20abstract.jpg)

The model configuration used in the paper is as following:

```
from histo_densevit import histo_densevit

model = histo_densevit \
        (image_size = (224,224), num_classes = 2,
        contextual_rpe = True,      # If True, use relative RPE, else use bias RPE
        contextual_input = 'k',   # Mode of relative RPE, 'k' indicates the relative RPE will be encoded
                                    # into keys
        deep_dense = True,          # If True, deploys dense connection
        )
```
