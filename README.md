### System environment
Unbuntu 16.04

### Preinstall Package

```bash 
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

### Create Conda Environment
```bash 
conda create --name tf-1.15 anaconda python=3.6
conda activate tf-1.15
```

### Install Tensorflow-1.15 GPU or CPU
```bash 
pip install tensorflow-gpu==1.15
```

or
```bash
pip install tensorflow==1.15
```

### Install Third-party Python Pakage
```bash
pip install gym
pip install graphviz
pip install pydotplus
pip install pyprind
pip install mpi4py
```

## TF2.19 Code-Only Migration

This codebase has been migrated from TensorFlow 1.15 to TensorFlow 2.19.0 (code-only migration).

### Migration Status
- ✅ All TF1.x patterns removed (sessions, placeholders, feed_dict)
- ✅ Graph2Seq encoder converted to Keras Layer
- ✅ PPO and MRLCO algorithms refactored with GradientTape
- ✅ Compatibility layers added for tf.contrib modules
- ✅ Checkpoint compatibility maintained (joblib format)

### Key Changes
1. **Eager Execution**: All code now uses TF2 eager execution style
2. **Keras Layers**: Policies and models use tf.keras.layers.Layer
3. **GradientTape**: Training loops use explicit gradient computation
4. **No Sessions**: Direct function calls replace session.run()

### Running with TF2.19
```bash
# Create new environment
conda create --name tf-2.19 python=3.8
conda activate tf-2.19

# Install TensorFlow 2.19
pip install tensorflow==2.19.0

# Install dependencies
pip install gym graphviz pydotplus pyprind mpi4py joblib
```

### Runtime Verification
**Note**: This migration was performed as code-only refactoring without execution. Runtime verification will occur on the server. Key areas to validate:
- Tensor shapes match expected values (see tests/golden/expected_shapes.json)
- Meta-learning gradient computation preserves first-order approximation
- Checkpoint loading from existing joblib files
- PPO loss computation and clipping behavior

### Compatibility Notes
- Existing checkpoints (joblib format) are supported via compat/checkpoint.py
- API interfaces remain unchanged for backward compatibility
- All tensor operations preserve original shapes and semantics
### Start Meta Training:
```bash
python meta_trainer.py
```
All the hyperparameters are defined in `meta_trainer.py` including the log file and save path of the trained model.

### Start Meta Evaluation:
After training, you will get the meta model. In order to fast adapt the meta model for new learning tasks in MEC, we need to conduct fine-tuning steps for the trained meta moodel.

```bash
python meta_evaluator.py
```

The training might take long time because of the large training set. All the training results and evaluation results can be found in the log file. 

Related paper: [Fast Adaptive Task Offloading in Edge Computing based on Meta Reinforcement Learning
](https://arxiv.org/abs/2008.02033)

If you like this research, please cite this paper:

```buildoutcfg
@article{wang2020fast,
  title={Fast Adaptive Task Offloading in Edge Computing Based on Meta Reinforcement Learning},
  author={Wang, Jin and Hu, Jia and Min, Geyong and Zomaya, Albert Y and Georgalas, Nektarios},
  journal={IEEE Transactions on Parallel and Distributed Systems},
  volume={32},
  number={1},
  pages={242--253},
  year={2020},
  publisher={IEEE}
}
```
