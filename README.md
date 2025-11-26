<div align="center">
<h1>Cosmos-inpainting: a controllable video model for pointcloud-based rendering videos</h1>

</div>

## Training
### Overview
We follow the standard Cosmos training pipeline, using its configuration system. Our key modifications are located in `cosmos_predict2/_src/predict2/inpainting`.
It includes the addition of datasets, conditioners, and models needed to post-train a new Cosmos version.

### Train
Run this command to lauch training:
```bash
sh 1shell_script/run_posttrain_inpaint.sh
```

For now, it only supports training on Libero dataset.

### Evalutaion
Run this command to evalute a checkpoint using the evaluation code in the trainer.

```bash
sh 1shell_script/run_eval_in_train_inpaint.sh
```

We need to do this because running a real evaluation during training causes us to run out of memory.
For now, we're handling evaluation manually by commenting out the `optimizer.step` in the training code.


Todos:
- [ ] Pre-compute text embeddings and avoid loading text encoder during training
- [ ] Check whether we can do evaluation in the training w/o text encoder
- [ ] Implement inference
- [ ] Support RoboTwin training
- [ ] Support multi-dataset training