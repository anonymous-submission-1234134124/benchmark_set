# How to run program repair benchmark on k8s

This tutorial provides an overview of how to run program repair benchmark on k8s

## CLI tools

This command executes these steps:
1) Builds docker image with your local code, and installs pip dependencies.
2) Tags the image with git hash, and pushes it to registry.
3) Creates Job manifest using configuration specified in the config passed
   with command line flag `--config`. Take a look at the sample config.
4) Applies the manifest to our cluster.

To build and push benchmark the image do:
```bash
python k8s/build_bench_image.py
```

The main issue is that you can apply the k8s manifest only from your laptop
when you are on VPN. Standalone GPU machines don't have access to the k8s
cluster. So you can build the image on standalone GPU machine, and then
Create the manifest, and apply it on your laptop. When you are building
the image, you need access to the code that you want to submit to k8s.

## Training Configuration Injection

**Important**: You no longer need to rebuild the Docker container when you only change training configuration parameters. The training config is now dynamically injected at runtime.

To use this feature:

1. Specify the training config file in your `config.yaml`:
```yaml
train_config_file: k8s/prog_repair_bench_config.yaml.yaml
```

2. The `submit_job.py` script will automatically:
   - Load your training config file (e.g., `k8s/prog_repair_bench_config.yaml.yaml`)
   - Process any OmegaConf interpolations in the config
   - Replace `TRAINING_CONFIG_PLACEHOLDER` with the actual config content
   - Submit the job with the dynamically generated command
