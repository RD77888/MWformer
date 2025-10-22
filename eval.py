import argparse
import json
import os
import sys
import warnings
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_provider.data_factory import data_provider
from model import MWformer

# 解决中文乱码
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

model_dict = {
    'MWformer': MWformer,
}


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params_m': total_params / 1e6,
        'trainable_params_m': trainable_params / 1e6
    }

def load_trained_model_and_args(checkpoint_path, args_path):
    with open(args_path, 'r', encoding='utf-8') as f:
        args_loaded_dict = json.load(f)
    args = type('Args', (), args_loaded_dict)
    if torch.cuda.is_available() and args_loaded_dict.get('use_gpu', True):
        args.device = torch.device(f'cuda:{args_loaded_dict.get("gpu", 0)}')
    else:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = torch.device("mps")
        else:
            args.device = torch.device("cpu")
    print(f"model device: {args.device}")
    model_name = args_loaded_dict.get('model', 'MWformer')
    candidate_keys = [
        model_name,
        model_name.lower(),
        model_name.capitalize()
    ]
    ModelModule = None
    for k in candidate_keys:
        if k in model_dict:
            ModelModule = model_dict[k]
            break

    Model = getattr(ModelModule, 'Model')
    model = Model(args).to(args.device)

    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    params = count_parameters(model)
    print(f"model param: {params['total_params_m']:.2f}M")
    return model, args, params

def calculate_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isinf(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    n_valid = len(y_true_clean)

    if n_valid == 0:
        return {k: np.nan for k in [
            'mae', 'mse', 'rmse', 'nse', 'kge',
            'mape', 'top2_mape', 'num_samples'
        ]}

    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    mean_obs = np.mean(y_true_clean)
    numerator_nse = np.sum((y_true_clean - y_pred_clean) ** 2)
    denominator_nse = np.sum((y_true_clean - mean_obs) ** 2)
    nse = 1 - (numerator_nse / denominator_nse) if denominator_nse != 0 else np.nan

    if n_valid < 2:
        r = np.nan
    else:
        r, _ = pearsonr(y_true_clean, y_pred_clean)

    obs_std = np.std(y_true_clean)
    pred_std = np.std(y_pred_clean)
    alpha = pred_std / obs_std if obs_std != 0 else np.nan
    beta = np.mean(y_pred_clean) / mean_obs if mean_obs != 0 else np.nan
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2) if not np.isnan(r) else np.nan
    eps = 1e-8
    y_true_nonzero = np.where(y_true_clean == 0, eps, y_true_clean)
    ape = np.abs((y_true_nonzero - y_pred_clean) / y_true_nonzero) * 100
    mape = np.mean(ape)
    ape_sorted = np.sort(ape)[::-1]
    top_n = max(1, int(n_valid * 0.02))
    top2_mape = np.mean(ape_sorted[:top_n])

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'nse': nse,
        'kge': kge,
        'mape': mape,
        'top2_mape': top2_mape,
        'num_samples': n_valid
    }

def parse_segments(segments_str, total_steps):
    segments = []
    for seg in segments_str.split():
        start_end = seg.split(':')
        if len(start_end) != 2:
            continue
        start = int(start_end[0])
        end = int(start_end[1])
        if start < 0 or end >= total_steps or start >= end:
            continue
        segments.append((start, end))
    return segments

def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")
    return directory

def simple_evaluation(model, test_loader, dataset_object, args, predict_segments):
    device = next(model.parameters()).device
    pred_len = args.pred_len
    print(f"pred_len: {pred_len}，pred section: {predict_segments}")
    all_predictions = []
    all_trues = []
    scaled_predictions = []
    scaled_trues = []
    all_timestamps = []
    total_inference_time = 0.0
    total_samples = 0
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        batch_runoff = batch_x[0].float().to(device)
        batch_rain = batch_x[1].float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
        batch_size = batch_runoff.shape[0]
        total_samples += batch_size

        with torch.no_grad():
            start_time = time.time()
            outputs, combined_pred = model(batch_runoff, batch_rain, batch_x_mark, batch_y, batch_y_mark)
            end_time = time.time()
            inference_time = end_time - start_time
            total_inference_time += inference_time
            final_pred = combined_pred[:, -pred_len:, -1:].squeeze(-1)
            true_tensor = batch_y[:, -pred_len:, -1:].squeeze(-1)

        scaled_pred_np = final_pred.detach().cpu().numpy()
        scaled_true_np = true_tensor.detach().cpu().numpy()
        pred_np = scaled_pred_np
        true_np = scaled_true_np

        for b in range(pred_np.shape[0]):
            sample_id = i * test_loader.batch_size + b
            scaled_predictions.append(pred_np[b])
            scaled_trues.append(true_np[b])
            pred_original = dataset_object.inverse_transform(pred_np[b])
            true_original = dataset_object.inverse_transform(true_np[b])
            all_predictions.append(pred_original)
            all_trues.append(true_original)

            pred_start_ts = dataset_object.get_pred_start_timestamp(sample_id)
            if pred_start_ts is not None:
                timestamps = pd.date_range(
                        start=pred_start_ts,
                        periods=pred_len,
                        freq=dataset_object.freq
                    )
                all_timestamps.append(timestamps)
            else:
                all_timestamps.append([pd.NaT] * pred_len)

    all_predictions = np.array(all_predictions)
    all_trues = np.array(all_trues)
    scaled_predictions = np.array(scaled_predictions)
    scaled_trues = np.array(scaled_trues)

    avg_inference_time_per_sample = total_inference_time / total_samples if total_samples > 0 else np.nan
    avg_inference_time_per_sample_ms = avg_inference_time_per_sample * 1000

    return {
        'overall': calculate_metrics(all_trues.flatten(), all_predictions.flatten()),
        'segments': [
            {
                'segment': f"{start}-{end}",
                'start': start,
                'end': end,
                'metrics': calculate_metrics(
                    all_trues[:, start:end + 1].flatten(),
                    all_predictions[:, start:end + 1].flatten()
                )
            } for (start, end) in predict_segments
        ],
        'all_trues': all_trues,
        'all_predictions': all_predictions,
        'scaled_overall': calculate_metrics(scaled_trues.flatten(), scaled_predictions.flatten()),
        'scaled_segments': [
            {
                'segment': f"{start}-{end}步",
                'start': start,
                'end': end,
                'metrics': calculate_metrics(
                    scaled_trues[:, start:end + 1].flatten(),
                    scaled_predictions[:, start:end + 1].flatten()
                )
            } for (start, end) in predict_segments
        ],
        'scaled_trues': scaled_trues,
        'scaled_predictions': scaled_predictions,
        'inference_metrics': {
            'total_inference_time_s': total_inference_time,
            'total_samples': total_samples,
            'avg_per_sample_ms': avg_inference_time_per_sample_ms
        },
        'all_timestamps': all_timestamps
    }


def export_simple_results(eval_results, output_dir, model_name, params):
    output_path = ensure_dir_exists(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_path, f"{model_name}_eval_{timestamp}")
    ensure_dir_exists(result_dir)

    overall_df = pd.DataFrame([eval_results['overall']])
    overall_df.to_csv(os.path.join(result_dir, 'overall_metrics.csv'), index=False, encoding='utf-8')

    seg_data = []
    for seg in eval_results['segments']:
        row = {'segment': seg['segment'], 'start': seg['start'], 'end': seg['end']}
        row.update(seg['metrics'])
        seg_data.append(row)
    seg_df = pd.DataFrame(seg_data)
    seg_df.to_csv(os.path.join(result_dir, 'segment_metrics.csv'), index=False, encoding='utf-8')

    # 修复：直接从eval_results根目录访问all_timestamps
    timestamps_flat = np.concatenate(
        [ts.astype(str) for ts in eval_results['all_timestamps']]
    ) if eval_results.get('all_timestamps') else np.array([])

    step_df = pd.DataFrame({
        'timestamp': timestamps_flat,  # 时间戳列
        'step': np.tile(np.arange(eval_results['all_trues'].shape[1]), eval_results['all_trues'].shape[0]),
        'true_value': eval_results['all_trues'].flatten(),
        'prediction': eval_results['all_predictions'].flatten()
    })
    step_df.to_csv(os.path.join(result_dir, 'stepwise_predictions.csv'), index=False, encoding='utf-8')

    if 'scaled_overall' in eval_results:
        scaled_overall_df = pd.DataFrame([eval_results['scaled_overall']])
        scaled_overall_df.to_csv(os.path.join(result_dir, 'scaled_overall_metrics.csv'),
                                 index=False, encoding='utf-8')

    if 'scaled_segments' in eval_results:
        scaled_seg_data = []
        for seg in eval_results['scaled_segments']:
            row = {'segment': seg['segment'], 'start': seg['start'], 'end': seg['end']}
            row.update(seg['metrics'])
            scaled_seg_data.append(row)
        scaled_seg_df = pd.DataFrame(scaled_seg_data)
        scaled_seg_df.to_csv(os.path.join(result_dir, 'scaled_segment_metrics.csv'),
                             index=False, encoding='utf-8')

    # 导出参数量指标
    params_df = pd.DataFrame([{
        'total_params_m': params['total_params_m'],
        'trainable_params_m': params['trainable_params_m'],
        'model_name': model_name
    }])
    params_df.to_csv(os.path.join(result_dir, 'model_parameters.csv'), index=False, encoding='utf-8')

    # 导出推理时间指标
    inference_df = pd.DataFrame([eval_results['inference_metrics']])
    inference_df.to_csv(os.path.join(result_dir, 'inference_time.csv'), index=False, encoding='utf-8')

    print(f"total_params: {params['total_params_m']:.2f}M")
    print(f"trainable_params_m: {params['trainable_params_m']:.2f}M")

    print(f"total_inference_time_s: {eval_results['inference_metrics']['total_inference_time_s']:.4f}s")
    print(f"total_samples: {eval_results['inference_metrics']['total_samples']}")

    print(overall_df[['mae', 'rmse', 'nse', 'kge', 'mape', 'top2_mape', 'num_samples']].to_string(index=False))

    print(seg_df[['segment', 'mae', 'rmse', 'nse', 'kge', 'mape', 'top2_mape', 'num_samples']].to_string(index=False))

    if 'scaled_overall' in eval_results:
        scaled_overall_df = pd.DataFrame([eval_results['scaled_overall']])
        print(
            scaled_overall_df[['mae', 'rmse', 'nse', 'kge', 'mape', 'top2_mape', 'num_samples']].to_string(index=False))

    if 'scaled_segments' in eval_results:
        scaled_seg_df = pd.DataFrame([
            {'segment': seg['segment'], **seg['metrics']} for seg in eval_results['scaled_segments']
        ])
        print(scaled_seg_df[['segment', 'mae', 'rmse', 'nse', 'kge', 'mape', 'top2_mape', 'num_samples']].to_string(
            index=False))

    return result_dir


def main():
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint path')
    parser.add_argument('--args_path', type=str, required=True, help='model args path')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--output_dir', type=str, default="./eval_results", help='output path')
    parser.add_argument('--predict_segments', type=str, required=True,
                        help='pred section("0:5 6:11 12:23")')
    args_cmd = parser.parse_args()

    if args_cmd.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args_cmd.gpu}')
        print(f"use GPU: {torch.cuda.get_device_name(args_cmd.gpu)}")
    else:
        device = torch.device('cpu')
        print("use CPU")

    try:
        model, args, params= load_trained_model_and_args(args_cmd.checkpoint, args_cmd.args_path)
        model.to(device)
        pred_len = args.pred_len

        predict_segments = parse_segments(args_cmd.predict_segments, pred_len)
        if not predict_segments:
            print("err:pred section is error")
            exit(1)

        test_data, test_loader = data_provider(args, flag='test')
        eval_results = simple_evaluation(model, test_loader, test_data, args, predict_segments)
        model_name = getattr(args, 'model', 'informer')

        result_dir = export_simple_results(eval_results, args_cmd.output_dir, model_name, params)
        print(f"\n result path: {result_dir}")

    except Exception as e:
        print(f"eval error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main()
