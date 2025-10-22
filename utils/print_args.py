def print_args(args):
    args_dict = vars(args)
    known_keys = {
        "Basic Config": ["task_name", "is_training", "model_id", "model"],
        "Data Loader": ["data", "root_path", "target", "freq", "checkpoints_last"],
        "Forecasting Task": ["seq_len", "label_len", "pred_len"],
        "Model Parameters": ["top_k", "num_kernels", "enc_in", "dec_in", "c_out", "d_model",
                             "n_heads", "e_layers", "d_layers", "d_ff", "moving_avg", "factor",
                             "distil", "dropout", "embed", "activation"],
        "Run Parameters": ["num_workers", "itr", "train_epochs", "batch_size", "patience",
                           "learning_rate", "des", "loss", "lradj", "use_amp"],
        "GPU": ["use_gpu", "gpu", "use_multi_gpu", "devices"],
        "De-stationary Projector Params": ["p_hidden_dims", "p_hidden_layers"]
    }

    def print_section(title, keys):
        print("\033[1m" + title + "\033[0m")
        for key in keys:
            value = args_dict.get(key, "N/A")
            if isinstance(value, list):  # Handle lists differently
                value = ', '.join(map(str, value))
            print(f'  {key + ":":<20}{str(value):<20}')
        print()

    # Print known sections
    for section, keys in known_keys.items():
        print_section(section, [k for k in keys if k in args_dict])

    # Find and print unknown arguments
    all_known_keys = set([item for sublist in known_keys.values() for item in sublist])
    unknown_keys = [k for k in args_dict.keys() if k not in all_known_keys]
    if unknown_keys:
        print("\033[1m" + "Additional Arguments" + "\033[0m")
        for key in unknown_keys:
            value = args_dict.get(key, "N/A")
            if isinstance(value, list):  # Handle lists differently
                value = ', '.join(map(str, value))
            print(f'  {key + ":":<20}{str(value):<20}')
