import os
import yaml
from src.pipeline import TimeSeriesPipeline
from src.utils.metrics import save_experiment_results


def main():
    config_path = "configs/config.yaml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config dosyası bulunamadı: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = config.get("experiment", {}).get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    pipeline = TimeSeriesPipeline(config_path)

    all_experiment_results = {}

    # ==================== 1. BATADAL Deneyleri ====================
    print("=" * 60)
    print("BATADAL - Temel Deneyler Başlıyor...")
    print("=" * 60)

    batadal_results = pipeline.run_batadal_pipeline()
    all_experiment_results["batadal_base"] = batadal_results

    if batadal_results:
        print("\nBATADAL Sonuçları:")
        for model_name, metrics in batadal_results.items():
            f1_info = metrics.get("f1_score", {})
            print(
                f"  {model_name}: F1 = {f1_info.get('mean', 0):.4f} ± {f1_info.get('std', 0):.4f}"
            )

    # ==================== 2. SKAB Deneyleri ====================
    print("\n" + "=" * 60)
    print("SKAB - GroupKFold Deneyleri Başlıyor...")
    print("=" * 60)

    skab_results = pipeline.run_skab_pipeline()
    all_experiment_results["skab_base"] = skab_results

    if skab_results:
        print("\nSKAB Sonuçları:")
        for model_name, metrics in skab_results.items():
            f1_info = metrics.get("f1_score", {})
            print(
                f"  {model_name}: F1 = {f1_info.get('mean', 0):.4f} ± {f1_info.get('std', 0):.4f}"
            )

    # ==================== 3. Gürültü Deneyleri ====================
    print("\n" + "=" * 60)
    print("BATADAL - Gürültü Deneyleri Başlıyor...")
    print("=" * 60)

    noise_results = pipeline.run_noise_experiment("batadal")
    all_experiment_results["batadal_noise"] = noise_results

    if noise_results:
        print("\nGürültülü BATADAL Sonuçları:")
        for model_name, metrics in noise_results.items():
            print(f"  {model_name}: F1 = {metrics.get('f1_score', 0):.4f}")

    # ==================== 4. Parametre Varyasyonu ====================
    print("\n" + "=" * 60)
    print("BATADAL - Parametre Varyasyonu Başlıyor...")
    print("=" * 60)

    param_results = pipeline.run_parameter_variation("batadal")
    all_experiment_results["batadal_param_variation"] = param_results

    if param_results:
        print("\nWindow Size Varyasyonu:")
        for ws, metrics in param_results.get("window_size", {}).items():
            print(
                f"  w={ws}: F1={metrics.get('f1_score', 0):.4f}, States={metrics.get('state_count', 0)}"
            )

        print("\nAlphabet Size Varyasyonu:")
        for als, metrics in param_results.get("alphabet_size", {}).items():
            print(
                f"  a={als}: F1={metrics.get('f1_score', 0):.4f}, States={metrics.get('state_count', 0)}"
            )

    # ==================== 5. Sonuçları Kaydet ====================
    save_experiment_results(
        all_experiment_results, os.path.join(output_dir, "all_experiment_results.json")
    )

    print("\n" + "=" * 60)
    print(f"Tüm deneyler tamamlandı. Sonuçlar '{output_dir}/' altına kaydedildi.")
    print("=" * 60)


if __name__ == "__main__":
    main()
