
# ////////////////////Utilizamos valores aun menores de C usando none para estudiar el cambio con este unico efecto, dado que el óptimo valor de C habia sido el menor con 0.3

grillas = {
    "logistica_estructural": {
        "modelo": modelos_base["logistica_estructural"],
        "params": {
            "clf__C": [0.3, 1.0, 3.0],
            "clf__class_weight": [None, "balanced"]
        }
    },

    "logistica_texto_sin_conteos": {
        "modelo": modelos_base["logistica_texto_sin_conteos"],
        "params": {
            #Anteriores: [0.3, 1.0, 3.0] 
            "clf__C": [0.01, 0.03, 0.1, 0.3, 1.0],
            "clf__class_weight": [None]
        }
    },

    "logistica_texto_conteos": {
        "modelo": modelos_base["logistica_texto_conteos"],
        "params": {
            #Anteriores: [0.3, 1.0, 3.0]
            "clf__C": [0.01, 0.03, 0.1, 0.3, 1.0],
            "clf__class_weight": [None]
        }
    },

    "arbol_estructural": {
        "modelo": modelos_base["arbol_estructural"],
        "params": {
            "clf__max_depth": [6, 10, 14, 18],
            "clf__min_samples_leaf": [50, 100, 200],
            "clf__class_weight": [None, "balanced"]
        }
    }
}

# ////////////////////////////En pipeline, agregamos más terminos 

def crear_preprocesamiento(incluir_texto=True, incluir_historial=True):
    numericas = NUMERICAS if incluir_historial else BASE_NUMERICAS

    transformadores = [
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False))
        ]), numericas),
        ("cat", crear_one_hot(), CATEGORICAS)
    ]

    if incluir_texto:
        transformadores.append(
            ("texto", TfidfVectorizer(
                #Probamos 20000 y 50000 dandonos mejor resultado 50000 (poco)
                max_features=50000,
                #Prueba con 3 y 5 frecuencias de documentos, nos quedamos en la prueba solo con 3 
                min_df=3,

                ngram_range=(1, 2),
                sublinear_tf=True
            ), "texto")
        )

    return ColumnTransformer(transformadores)

# ///////////////////////En preparacion de caracteristicas, agregamos categoria

def crear_features_base(df):
    X = df.copy()

    X["pub_fecha"] = pd.to_datetime(
        X["pub_fecha"],
        errors="coerce",
        utc=True
    )

    #Originalmente se tiene: título + descripción.
    X["texto"] = (
        X["pub_titulo"].fillna("").astype(str) + " " +
        X["pub_descripcion"].fillna("").astype(str)
    )

    X["titulo_largo"] = (
        X["pub_titulo"].fillna("").astype(str).str.len()
    )
    X["descripcion_larga"] = (
        X["pub_descripcion"].fillna("").astype(str).str.len()
    )
    X["texto_palabras"] = X["texto"].str.split().str.len()

    #Agregamos categoria quedandoa: título + descripción + categoría.
    X["texto"] = (
        X["texto"] + " " +
        X["pub_categoria"].fillna("").astype(str)
    )

    X["pub_monto_estimado"] = pd.to_numeric(
        X["pub_monto_estimado"],
        errors="coerce"
    )
    X["monto_nulo"] = X["pub_monto_estimado"].isna().astype(int)
    X["moneda_nula"] = X["pub_moneda"].isna().astype(int)

    X["mes_publicacion"] = X["pub_fecha"].dt.month
    X["dia_semana_publicacion"] = X["pub_fecha"].dt.dayofweek

    categoria = X["pub_categoria"].fillna("desconocido").astype(str)

    X["categoria_principal"] = (
        categoria.str.split("/").str[0].str.strip()
    )
    X["cat_niveles"] = categoria.str.count("/") + 1

    for col in [
        "pub_n_items",
        "pub_cantidad_total",
        "pub_n_categorias",
        "pub_dias_postulacion",
        "pub_dias_consultas"
    ]:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in CATEGORICAS:
        X[col] = X[col].fillna("desconocido").astype(str)

    X["comprador_key"] = normalizar_llave(X["comprador_id"])
    X["categoria_key"] = normalizar_llave(X["pub_categoria_id"])
    X["comprador_categoria_key"] = (
        X["comprador_key"] + "|" + X["categoria_key"]
    )

    return X


#Se ejecutó nuevamente para elegir el umbral con los cambios al modelo entrenado con los cambios

mejor_threshold, f1_umbral, precision_umbral, recall_umbral = (
  mejor_umbral_f1(y_val_umbral, proba_umbral)
)
