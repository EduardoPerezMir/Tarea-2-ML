"""Reproduce la ablación A/B de pregunta2.ipynb sin evaluar C, D ni Kaggle.

Ejecutar desde esta carpeta: python comparar_proporcion_consultas.py
La configuración original queda congelada a partir del ganador registrado antes
de añadir la proporción. No se vuelve a seleccionar un modelo por variante.
Las funciones, los pipelines y la celda de comparación se leen del notebook.
"""

import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path


def main():
    notebook_path = Path("pregunta2.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    def source_with(marker):
        return next("".join(c["source"]) for c in cells if marker in "".join(c["source"]))

    namespace = {}

    def execute(source, predicate=None):
        tree = ast.parse(source)
        if predicate is not None:
            tree.body = [node for node in tree.body if predicate(node)]
        exec(compile(tree, str(notebook_path), "exec"), namespace)

    execute(source_with("import numpy as np"))
    pd, np = namespace["pd"], namespace["np"]
    seed = namespace["SEED"]
    target = namespace["TARGET"]
    corte_a = pd.Timestamp("2024-08-01", tz="UTC")
    corte_b = pd.Timestamp("2024-10-01", tz="UTC")
    datos = pd.read_csv("train.csv")
    datos["pub_fecha"] = pd.to_datetime(datos["pub_fecha"], errors="raise", utc=True)
    # Fuera de A/B no se construyen features ni se calculan métricas.
    datos = datos.loc[datos["pub_fecha"] < corte_b].copy()
    train = datos.loc[datos["pub_fecha"] < corte_a].copy()
    validacion = datos.loc[datos["pub_fecha"] >= corte_a].copy()
    assert train["id"].is_unique and validacion["id"].is_unique
    assert set(train["id"]).isdisjoint(validacion["id"])
    assert train["pub_fecha"].max() < validacion["pub_fecha"].min()
    namespace.update({
        "datos": datos, "train_modelos": train, "val_hparam": validacion,
        "VENTANAS_AJUSTE": {"A": (train.index, corte_a)}, "registro_ajustes": [],
    })
    # Costos definidos solo con entrenamiento; no afectan la selección por AP/F1.
    execute(source_with("def definir_costos_desbalance("))

    def is_feature_definition(node):
        if isinstance(node, ast.FunctionDef):
            return True
        return isinstance(node, ast.Assign) and not any(
            isinstance(t, ast.Name) and t.id.startswith(("X_", "y_")) for t in node.targets
        )

    execute(source_with("def crear_features_base("), is_feature_definition)
    execute(source_with("def crear_preprocesamiento("))
    construir = namespace["construir_features"]
    namespace.update({
        "X_train_modelos": construir(train),
        "X_val_hparam": construir(validacion, referencia=train),
        "y_train_modelos": train[target].astype(int),
        "y_val_hparam": validacion[target].astype(int),
    })

    # Reutiliza las pruebas nuevas de casos límite que están en el notebook.
    acceptance_tree = ast.parse(source_with("def ejecutar_pruebas_aceptacion("))
    acceptance = next(n for n in acceptance_tree.body if isinstance(n, ast.FunctionDef))
    for position, statement in enumerate(acceptance.body):
        if isinstance(statement, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "muestra" for t in statement.targets
        ):
            acceptance.body = acceptance.body[:position]
            break
    acceptance_tree.body = [acceptance]
    exec(compile(acceptance_tree, str(notebook_path), "exec"), namespace)
    namespace["ejecutar_pruebas_aceptacion"]()
    muestra = train.head(100).copy()
    alterada = muestra.copy()
    alterada[target] = np.random.default_rng(seed).integers(0, 2, len(muestra))
    pd.testing.assert_frame_equal(construir(muestra), construir(alterada))
    # Añadir publicaciones posteriores de B tampoco altera las features de A.
    pd.testing.assert_frame_equal(
        construir(muestra), construir(pd.concat([muestra, validacion.head(50)])).loc[muestra.index]
    )
    print("Casos límite y ausencia de dependencia del objetivo/futuro: OK", flush=True)

    def is_model_definition(node):
        if isinstance(node, (ast.FunctionDef, ast.For)):
            return True
        return isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in {"modelos_base", "modelos_base_originales"}
            for t in node.targets
        )

    execute(source_with("def metricas_con_umbral("), is_model_definition)
    # Ganador original guardado en las salidas previas de la búsqueda en B.
    # Son los mismos hiperparámetros para ambas variantes, sin una nueva búsqueda.
    namespace["resultados_busqueda"] = [{
        "modelo": "logistica_texto_conteos",
        "params": {"clf__C": 0.3, "clf__class_weight": None},
        "ap": 0.33799307907204645,
    }]
    captured_outputs = []

    def display_capture(obj):
        text_plain = obj.to_string() if hasattr(obj, "to_string") else str(obj)
        if hasattr(obj, "data") and hasattr(obj.data, "to_string"):
            text_plain = obj.data.to_string(float_format=lambda value: f"{value:.8f}")
        data = {"text/plain": [text_plain]}
        if hasattr(obj, "to_html"):
            data["text/html"] = [obj.to_html()]
        captured_outputs.append({"output_type": "display_data", "metadata": {}, "data": data})

    namespace["display"] = display_capture
    comparison_cell = next(c for c in cells if c["cell_type"] == "code" and
                           "UMBRAL_COMPARACION_B = 0.5" in "".join(c["source"]))
    print("Ajustando ambas variantes con A; evaluando solo B...", flush=True)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        execute("".join(comparison_cell["source"]))
    if stdout.getvalue():
        captured_outputs.append({"output_type": "stream", "name": "stdout", "text": stdout.getvalue().splitlines(True)})
    table = namespace["tabla_comparacion_B"]
    print(table.to_string(float_format=lambda value: f"{value:.8f}"), flush=True)
    print(stdout.getvalue(), end="", flush=True)

    # El original reproducido coincide con el resultado previo, sin consultar D.
    assert np.isclose(table.loc["original", "ap"], 0.33799307907204645, atol=1e-12, rtol=0)
    assert np.isclose(table.loc["original", "f1"], 0.07771260997067449, atol=1e-12, rtol=0)
    # Releer antes de escribir para preservar ediciones hechas durante el ajuste.
    current_notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    current_cell = next(c for c in current_notebook["cells"] if c["cell_type"] == "code" and
                        "UMBRAL_COMPARACION_B = 0.5" in "".join(c["source"]))
    assert current_cell["source"] == comparison_cell["source"], "La celda cambió durante la ejecución."
    current_cell["outputs"] = captured_outputs
    current_cell["execution_count"] = 1
    notebook_path.write_text(json.dumps(current_notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    result = {
        "versiones": {"numpy": np.__version__, "pandas": pd.__version__,
                      "scikit-learn": namespace["sklearn"].__version__},
        "seed": seed, "modelo": namespace["nombre_comparacion"],
        "parametros": namespace["params_comparacion"], "umbral": namespace["UMBRAL_COMPARACION_B"],
        "origen_configuracion": "Ganador original en B, congelado antes de añadir la proporción",
        "bloques": {
            label: {"n": len(df), "desde": df["pub_fecha"].min().isoformat(),
                    "hasta": df["pub_fecha"].max().isoformat(),
                    "ids_sha256": hashlib.sha256(df["id"].to_numpy(dtype="<i8").tobytes()).hexdigest()}
            for label, df in (("A", train), ("B", validacion))
        },
        "nulos_proporcion": {
            "A": int(namespace["X_train_modelos"]["proporcion_consultas"].isna().sum()),
            "B": int(namespace["X_val_hparam"]["proporcion_consultas"].isna().sum()),
        },
        "metricas_B": table.to_dict(orient="index"),
        "matrices_confusion_B": {
            variant: metrics["matriz_confusion"].tolist()
            for variant, metrics in namespace["metricas_comparacion"].items()
        },
        "conservar_proporcion": namespace["CONSERVAR_PROPORCION"],
        "limitacion": "B se reutiliza; esta ablación no demuestra generalización ni consistencia temporal.",
    }
    Path("comparacion_proporcion_consultas_B.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
