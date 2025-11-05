import os
import math
import optuna
import pathlib
import pickle
import mlflow
import pandas as pd
from dotenv import load_dotenv
from optuna.samplers import TPESampler
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient # Necesario para la gestión de alias
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from prefect import flow, task

# --- CONSTANTES ---
# Nombre del modelo en Unity Catalog
MODEL_REGISTRY_NAME = "workspace.default.nyc-taxi-model-prefect"
EXPERIMENT_NAME = "/Users/danydelatorregallo@gmail.com/nyc-taxi-experiment-prefect"

# --------------------------
# TAREAS DE INGESTA Y PRE-PROCESAMIENTO
# --------------------------

@task(name="Read Data")
def read_data(file_path: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    # ... (código sin cambios)
    df = pd.read_parquet(file_path)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df

@task(name="Add Features")
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """Add features to the model"""
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"] 
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, X_val, y_train, y_val, dv

# --------------------------
# TAREA DE HYPER-PARÁMETROS
# --------------------------

@task(name="Hyperparameter Tunning")
def hyper_parameter_tunning(X_train, X_val, y_train, y_val, dv, model_obj):
    
    mlflow.sklearn.autolog()
    
    # CORRECCIÓN 1: Usar isinstance() para comparar tipos de objetos
    if isinstance(model_obj, RandomForestRegressor):
        model_family = "random_forest_regressor"
        model_name = "Random Forest Regressor"
        
        def objective(trial: optuna.trial.Trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 5, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_float("max_features", 0.1, 1.0, log=False),
                "random_state": 42,
                "n_jobs": -1,
            }

            with mlflow.start_run(nested=True):
                mlflow.set_tag("model_family", model_family)
                mlflow.log_params(params)
                
                model = model_obj.__class__(**params)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_val)
                rmse = root_mean_squared_error(y_val, y_pred)
                mlflow.log_metric("rmse", rmse)
                
                signature = infer_signature(X_val, y_pred)
                mlflow.sklearn.log_model(model, name="model", input_example=X_val[:5], signature=signature)
            return rmse

    elif isinstance(model_obj, GradientBoostingRegressor):
        model_family = "gradient_boosting_regressor"
        model_name = "Gradient Boosting Regressor"

        def objective(trial: optuna.trial.Trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "max_features": trial.suggest_categorical("max_features", ['sqrt', 'log2', None]),
                "random_state": 42,
            }

            with mlflow.start_run(nested=True):
                mlflow.set_tag("model_family", model_family)
                mlflow.log_params(params)
                
                model = model_obj.__class__(**params)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_val)
                rmse = root_mean_squared_error(y_val, y_pred)
                mlflow.log_metric("rmse", rmse)
                
                signature = infer_signature(X_val, y_pred)
                mlflow.sklearn.log_model(model, name="model", input_example=X_val[:5], signature=signature)
            return rmse
    else:
        raise ValueError(f"Modelo no soportado para Optuna: {type(model_obj)}")

    # CORRECCIÓN 2: Ejecución fuera de la condición IF para solucionar UnboundLocalError
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    
    with mlflow.start_run(run_name=f"{model_name} Hyperparameter Optimization (Optuna)", nested=True):
        study.optimize(objective, n_trials=3)

    best_params = study.best_params
    
    if "max_depth" in best_params:
        best_params["max_depth"] = int(best_params["max_depth"])
    
    best_params["random_state"] = 42

    return best_params

# --------------------------
# TAREA DE ENTRENAMIENTO FINAL
# --------------------------

@task(name="Train Best Model")
def train_best_model(X_train, X_val, y_train, y_val, dv, best_params, model_obj): # Se eliminó -> None
    """train a model with best hyperparams and writes artifacts to MLflow, returning the run_id."""

    if isinstance(model_obj, GradientBoostingRegressor):
        model_family = "Gradient Boosting Regressor"
    elif isinstance(model_obj, RandomForestRegressor):
        model_family = "Random Forest Regressor"
    else:
        raise ValueError("Modelo no reconocido.")

    # El run final se abre y se cierra en el bloque 'with', permitiendo capturar el run_id.
    with mlflow.start_run(run_name=f"{model_family} Final Model") as run:
        mlflow.log_params(best_params)

        mlflow.set_tags({
            "project": "NYC Taxi Time Prediction Project",
            "optimizer_engine": "optuna",
            "model_family": model_family,
            "feature_set_version": 1,
        })

        # Entrenar el modelo FINAL
        model = model_obj.__class__(**best_params)
        model.fit(X_train, y_train)

        # Evaluar y registrar la métrica final
        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        # Guardar artefactos
        pathlib.Path("preprocessor").mkdir(exist_ok=True)
        with open("preprocessor/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("preprocessor/preprocessor.b", artifact_path="preprocessor")

        # Preparar y guardar la signature y el modelo
        feature_names = dv.get_feature_names_out()
        input_example = pd.DataFrame(X_val[:5].toarray(), columns=feature_names)
        signature = infer_signature(input_example, y_val[:5])

        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=input_example,
            signature=signature,
        )
    
    # Retornar el ID del run para el registro en Unity Catalog
    return run.info.run_id

# --------------------------
# TAREA DE REGISTRO DE VERSIÓN
# --------------------------

@task(name= "Register Model Version")
def register_model_version(run_id: str, model_name: str) -> None:
    """Registra la versión del modelo en Unity Catalog usando el run_id."""
    model_uri_full = f"runs:/{run_id}/model"
    
    # Se registra el modelo como una nueva versión en Unity Catalog
    mlflow.register_model(
        model_uri=model_uri_full,
        name=model_name)
    print(f"✅ Modelo registrado como nueva versión de '{model_name}' usando run ID: {run_id}")

# --------------------------
# TAREA DE CHAMPION/CHALLENGER (NUEVA LÓGICA)
# --------------------------

@task(name="Manage Model Aliases")
def manage_model_alias(model_name: str) -> None:
    client = MlflowClient()
    
    # 1. Obtener todas las versiones del modelo registrado en Unity Catalog
    try:
        # CORRECCIÓN: Usar el parámetro filter_string y asegurarse de que la función se llama correctamente.
        all_versions = client.search_model_versions(filter_string=f"name='{model_name}'")
    except Exception as e:
        # ... (resto del manejo de errores)
        print(f"ERROR: No se pudo encontrar el modelo '{model_name}' en el registro. Error: {e}")

    scored_versions = []
    
    # 2. Recorrer las versiones, obtener el RMSE del run asociado
    for version in all_versions:
        run_id = version.run_id
        if not run_id:
            continue
            
        try:
            # Obtener el run para extraer la métrica RMSE
            run_data = client.get_run(run_id)
            # Buscamos 'rmse' que es la métrica registrada en el run final
            rmse = run_data.data.metrics.get("rmse") 
            
            if rmse is not None:
                scored_versions.append({
                    "version": version.version,
                    "rmse": rmse,
                    "aliases": version.aliases,
                    "model_family": run_data.data.tags.get("model_family", "N/A"),
                })
        except Exception:
            # Ignoramos versiones que no tienen el run asociado disponible o métrica.
            pass

    if len(scored_versions) < 2:
        print(f"ADVERTENCIA: Solo se encontraron {len(scored_versions)} versiones válidas con métrica 'rmse'. Se requieren al menos 2.")
        return

    # 3. Ordenar por RMSE (ascendente: el más bajo es el mejor)
    scored_versions.sort(key=lambda x: x["rmse"])

    champion = scored_versions[0]
    challenger = scored_versions[1]
    
    print(f"🏅 Asignando CHAMPION: Versión {champion['version']}...")
    # CAMBIO CLAVE AQUÍ: set_registered_model_alias
    client.set_registered_model_alias(
        name=model_name, 
        alias="Champion", 
        version=champion["version"]
    )

    print(f"🚀 Asignando CHALLENGER: Versión {challenger['version']}...")
    # CAMBIO CLAVE AQUÍ: set_registered_model_alias
    client.set_registered_model_alias(
        name=model_name, 
        alias="Challenger", 
        version=challenger["version"]
    )

    print(f"Champion y Challenger asignados exitosamente.")

# --------------------------
# FLUJO PRINCIPAL
# --------------------------

@flow(name="Main Flow")
def main_flow(year: int, month_train: str, month_val: str) -> None:
    """The main training pipeline for competitive model selection."""
    
    train_path = f"../data/green_tripdata_{year}-{month_train}.parquet"
    val_path = f"../data/green_tripdata_{year}-{month_val}.parquet"
    
    load_dotenv(override=True)
    
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

    # 1. Load y Transformación de Datos
    df_train = read_data(train_path)
    df_val = read_data(val_path)

    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)
    
    # --- Ejecutar Modelos Competidores ---
    
    # 2. Entrenamiento de RANDOM FOREST
    rf_model = RandomForestRegressor(random_state=42)
    rf_best_params = hyper_parameter_tunning(X_train, X_val, y_train, y_val, dv, rf_model)
    # Entrenar y obtener el run_id del modelo final
    rf_run_id = train_best_model(X_train, X_val, y_train, y_val, dv, rf_best_params, rf_model)
    register_model_version(rf_run_id, MODEL_REGISTRY_NAME) # <-- Registro de Versión RF

    # 3. Entrenamiento de GRADIENT BOOSTING
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_best_params = hyper_parameter_tunning(X_train, X_val, y_train, y_val, dv, gb_model)
    # Entrenar y obtener el run_id del modelo final
    gb_run_id = train_best_model(X_train, X_val, y_train, y_val, dv, gb_best_params, gb_model)
    register_model_version(gb_run_id, MODEL_REGISTRY_NAME) # <-- Registro de Versión GB
    
    # 4. Asignar Champion/Challenger (Compara TODAS las versiones existentes)
    manage_model_alias(MODEL_REGISTRY_NAME)

# --- EJECUCIÓN ---
if __name__ == "__main__":
    main_flow(year=2025, month_train="01", month_val="02")