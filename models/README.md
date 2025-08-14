
```

torch.onnx.export(
        model=model,
        args=model.example_inputs_for_export(feat_size, device=gpu_device),
        f=os.path.join(ckpt_dir, "model.onnx"),
        input_names=["feature", "length"],
        output_names=["probs"],
        dynamic_axes= {
            "feature": {0: "batch"},
            "length": {0: "batch"},
            "probs": {0: "batch"}
        })
```