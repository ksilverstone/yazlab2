import json

def save_explainability_log(results: list, output_path: str):
    """
    Pipeline'dan dönen açıklanabilirlik log listesini JSON formatında okunabilir (indent=4) şekilde kaydeder.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
